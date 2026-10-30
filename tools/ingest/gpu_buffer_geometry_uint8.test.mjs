import assert from 'node:assert/strict';
import test from 'node:test';
import {BufferAttribute, BufferGeometry} from '../../upstream/three.js/build/three.core.js';
import {GLTFLoader} from '../../upstream/three.js/examples/jsm/loaders/GLTFLoader.js';
import {createGpuBufferGeometry, bufferGeometrySnapshot, createGpuProgramGeometry, programGeometrySnapshot} from './gpu_buffer_geometry.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {byteIndexedGLB} from './fixtures/gltf_byte_indices.mjs';

const geometry = (values = [0, 1, 2, 0, 2, 3]) => new BufferGeometry()
  .setAttribute('position', new BufferAttribute(new Float32Array(12), 3))
  .setIndex(new BufferAttribute(new Uint8Array(values), 1));
const indices = snapshot => [...new Uint32Array(snapshot.indexBuffer.data)].slice(0, snapshot.indexCount);

test('actual GLTFLoader byte-indexed GLB enters native geometry without replacing its accessor', async () => {
  const asset = await new GLTFLoader().parseAsync(byteIndexedGLB(), ''), g = asset.scene.children[0].geometry;
  const source = g.index, array = source.array, d = geometryDevice();
  assert.ok(array instanceof Uint8Array); assert.equal(g.attributes.color.normalized, true);
  const gpu = createGpuBufferGeometry(d, g), snapshot = bufferGeometrySnapshot(gpu, d);
  assert.equal(snapshot.indexFormat, 'uint32'); assert.deepEqual(indices(snapshot), [0, 1, 2, 0, 2, 3]);
  assert.equal(g.index, source); assert.equal(source.array, array);
  assert.equal(gpu.bufferBytes, 48 + 64 + 24, 'budget includes widened indices and normalized vertex color');
  await gpu.whenIdle(); gpu.dispose(); assert.ok(d.buffers.every(buffer => buffer.destroyed));
});

test('byte-index partial updates preserve element ranges, stale bytes and callback ordering', () => {
  const g = geometry(), index = g.index, source = index.array, d = geometryDevice(), calls = [];
  index.addUpdateRange(4, 1); index.onUpload(function () { calls.push(this.updateRanges.map(range => ({...range}))); });
  const gpu = createGpuBufferGeometry(d, g), snapshot = () => bufferGeometrySnapshot(gpu, d);
  assert.deepEqual(calls, [[{start: 4, count: 1}]], 'first upload retains source ranges');
  index.clearUpdateRanges(); source.set([3, 2, 1, 2, 1, 0]); gpu.update();
  assert.deepEqual(indices(snapshot()), [0, 1, 2, 0, 2, 3], 'unversioned edits remain stale');
  index.addUpdateRange(1, 1); index.addUpdateRange(3, 1); index.needsUpdate = true; gpu.update();
  assert.deepEqual(indices(snapshot()), [0, 2, 1, 2, 2, 3], 'source +1 coalescing includes only its intervening element');
  assert.deepEqual(calls.at(-1), []); assert.equal(d.writes.at(-1).offset, 4);
  assert.equal(d.writes.at(-1).input.length, 12); assert.equal(index.array, source);
  index.addUpdateRange(4, 0); index.needsUpdate = true; gpu.update();
  assert.deepEqual(indices(snapshot()), [0, 2, 1, 2, 1, 0], 'zero-length source range uploads its remainder');
  gpu.dispose();
});

for (const attributeTypes of ['webgl', 'webgpu']) test(`byte index 255 follows the ${attributeTypes} primitive-restart contract`, () => {
  const d = geometryDevice(), g = geometry([0, 1, 255, 2, 3]), source = g.index.array;
  const gpu = createGpuProgramGeometry(d, g, [], {attributeTypes}), snapshot = programGeometrySnapshot(gpu, d);
  assert.equal(snapshot.indexFormat, 'uint32');
  assert.deepEqual(indices(snapshot), [0, 1, attributeTypes === 'webgl' ? 0xffffffff : 255, 2, 3]);
  assert.equal(snapshot.readIndex(2), attributeTypes === 'webgl' ? 0xffffffff : 255);
  assert.equal(g.index.array, source); assert.equal(source[2], 255); gpu.dispose();
});

test('widened index residency budgets are admitted before allocation, writes and source callbacks', () => {
  const d = geometryDevice(), g = geometry(); let calls = 0; g.index.onUpload(() => calls++);
  assert.throws(() => createGpuBufferGeometry(d, g, {maxBytes: 48 + 23}), {code: 'GEOMETRY_GPU_LIMIT'});
  assert.equal(d.buffers.length, 0); assert.equal(d.writes.length, 0); assert.equal(calls, 0);
  const gpu = createGpuBufferGeometry(d, g, {maxBytes: 48 + 24}); assert.equal(gpu.bufferBytes, 72); gpu.dispose();
});

test('byte index histories survive callback failure, array release and source disposal', () => {
  const d = geometryDevice(), g = geometry(), gpu = createGpuBufferGeometry(d, g), index = g.index;
  const error = Error('index callback'); index.array[2] = 3; index.addUpdateRange(2, 1); index.needsUpdate = true;
  index.onUpload(() => { throw error; }); assert.throws(() => gpu.update(), value => value === error);
  const before = bufferGeometrySnapshot(gpu, d);
  assert.equal(before.readIndex(2), 3); assert.equal(gpu.failed, false); assert.equal(index.updateRanges.length, 0);
  index.onUpload(() => { index.array = null; }); gpu.update(); gpu.update();
  assert.deepEqual(indices(bufferGeometrySnapshot(gpu, d)), [0, 1, 3, 0, 2, 3]);
  g.dispose(); assert.equal(before.indexBuffer.destroyed, true); assert.equal(gpu.bufferBytes, 0);
  assert.throws(() => gpu.update(), {code: 'GEOMETRY_GPU_STORAGE'});
  g.setIndex(new BufferAttribute(new Uint8Array([3, 2, 1]), 1)); gpu.update();
  assert.deepEqual(indices(bufferGeometrySnapshot(gpu, d)), [3, 2, 1]); gpu.dispose();
});

test('byte index growth and type changes retain existing replacement requirements', () => {
  const d = geometryDevice(), g = geometry(), gpu = createGpuBufferGeometry(d, g), writes = d.writes.length;
  g.index.array = new Uint8Array(9); g.index.count = 9; g.index.needsUpdate = true;
  assert.throws(() => gpu.update(), {code: 'GEOMETRY_GPU_RESIZE'}); assert.equal(d.writes.length, writes);
  g.index.array = new Uint16Array(6); g.index.count = 6;
  assert.throws(() => gpu.update(), {code: 'GEOMETRY_GPU_FORMAT'}); assert.equal(d.writes.length, writes);
  g.setIndex(new BufferAttribute(new Uint16Array([0, 1, 2]), 1)); gpu.update();
  assert.equal(bufferGeometrySnapshot(gpu, d).indexFormat, 'uint16'); gpu.dispose();
});
