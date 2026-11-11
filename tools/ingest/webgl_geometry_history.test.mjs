import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.module.js';
import {WebGLAttributes} from '../../upstream/three.js/src/renderers/webgl/WebGLAttributes.js';
import {createWebGLInstanceHistory} from './webgl_instance_history.mjs';
import {createThreePassInstances} from './three_pass_instances.mjs';
import {captureThreeMaterialState} from './three_material_snapshot.mjs';
import {createGpuBufferGeometry, bufferGeometrySnapshot} from './gpu_buffer_geometry.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

function oracle() {
  let bound;
  return WebGLAttributes({createBuffer: () => ({}), bindBuffer(_target, buffer) { bound = buffer; },
    bufferData(_target, array) { bound.bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength).slice(); },
    bufferSubData(_target, offset, array, start = 0, count = 0) {
      const width = array.BYTES_PER_ELEMENT;
      bound.bytes.set(new Uint8Array(array.buffer, array.byteOffset + width * start, width * (count || array.length - start)), offset);
    }, deleteBuffer() {}});
}

for (const Type of [Float32Array, Int16Array, Uint8Array, Uint16Array, Uint32Array])
test(`raw ${Type.name} geometry history matches pinned WebGLAttributes upload ranges`, () => {
  const history = createWebGLInstanceHistory({geometry: true});
  const source = new T.BufferAttribute(new Type([0, 1, 2, 3, 4, 5]), 1), reference = source.clone(), gl = oracle();
  const compare = () => {
    gl.update(reference, 0);
    const array = history.capture(source);
    assert.deepEqual(new Uint8Array(array.buffer, array.byteOffset, array.byteLength), gl.get(reference).buffer.bytes);
    assert.deepEqual(source.updateRanges, reference.updateRanges);
  };
  // Initial upload leaves ranges intact; a versionless edit stays CPU-only.
  source.addUpdateRange(1, 1); reference.addUpdateRange(1, 1); compare();
  source.array[0] = reference.array[0] = 9; compare();
  const sourceRange = source.updateRanges[0], referenceRange = reference.updateRanges[0];
  for (const attribute of [source, reference]) {
    attribute.array[1] = 7; attribute.array[2] = 8; attribute.array[3] = 6;
    attribute.addUpdateRange(3, 1); attribute.needsUpdate = true;
  }
  compare(); assert.deepEqual(sourceRange, referenceRange); assert.deepEqual(sourceRange, {start: 1, count: 3});
  for (const attribute of [source, reference]) {
    attribute.array[2] = 9; attribute.array[5] = 10; attribute.addUpdateRange(2, 0); attribute.needsUpdate = true;
  }
  compare(); history.dispose();
});

test('live geometry owners and queued history share normalized colors, byte indices and upload callbacks', () => {
  const device = geometryDevice(), history = createWebGLInstanceHistory({geometry: true}), geometry = new T.BufferGeometry();
  const position = new T.BufferAttribute(new Float32Array([-1, -1, 0, 1, -1, 0, 0, 1, 0]), 3);
  const color = new T.BufferAttribute(new Uint8Array([255, 128, 0, 255, 128, 0, 255, 128, 0]), 3, true);
  const index = new T.BufferAttribute(new Uint8Array([0, 1, 2]), 1);
  geometry.setAttribute('position', position); geometry.setAttribute('color', color); geometry.setIndex(index);
  const calls = []; for (const attribute of [position, color, index]) attribute.onUpload(function () { assert.equal(this, attribute); calls.push(attribute); });
  const first = createGpuBufferGeometry(device, geometry, {instanceHistory: history});
  assert.deepEqual(calls, [position, color, index]);
  color.array[0] = 0; color.array[1] = 64; color.addUpdateRange(1, 1); color.needsUpdate = true;
  index.array[0] = 2; index.array[1] = 2; index.addUpdateRange(1, 1); index.needsUpdate = true;
  history.capture(color); history.capture(index);
  const second = createGpuBufferGeometry(device, geometry, {instanceHistory: history}); first.update();
  assert.deepEqual(calls, [position, color, index, color, index], 'replica creation restores bytes without repeating source effects');
  for (const gpu of [first, second]) {
    const snapshot = bufferGeometrySnapshot(gpu, device);
    assert.deepEqual([...new Float32Array(snapshot.vertexBuffers[1].data)].slice(0, 3), [1, Math.fround(64 / 255), 0]);
    assert.deepEqual([...new Uint32Array(snapshot.indexBuffer.data)].slice(0, 3), [0, 2, 2]);
  }
  first.dispose(); second.dispose(); geometry.dispose(); history.dispose();
});

test('queued geometry byte limits refuse before acknowledging updates and recover after release', () => {
  const owner = createThreePassInstances(T, {maxBytes: 48}), scene = new T.Scene(), camera = new T.PerspectiveCamera();
  const geometry = new T.BufferGeometry().setAttribute('position', new T.BufferAttribute(new Float32Array(9), 3));
  const mesh = new T.Mesh(geometry, new T.MeshBasicMaterial()); mesh.frustumCulled = false; scene.add(mesh);
  const frame = () => captureThreeMaterialState(T, scene, {camera, selectionOnly: true, allowInstances: true});
  const first = owner.capture(scene, camera, frame());
  geometry.attributes.position.addUpdateRange(0, 1); geometry.attributes.position.needsUpdate = true;
  assert.throws(() => owner.capture(scene, camera, frame()), {code: 'F3D_RENDERER_UNSUPPORTED'});
  assert.deepEqual(geometry.attributes.position.updateRanges, [{start: 0, count: 1}]);
  assert.equal(owner.pendingBytes, 36); first.release();
  const next = owner.capture(scene, camera, frame()); assert.deepEqual(geometry.attributes.position.updateRanges, []);
  next.release(); assert.equal(owner.pendingBytes, 0); owner.dispose(); mesh.material.dispose(); geometry.dispose();
});

test('disposing a geometry preserves replaced attributes still shared by another geometry', () => {
  const device = geometryDevice(), history = createWebGLInstanceHistory({geometry: true}), gl = oracle();
  const shared = new T.BufferAttribute(new Float32Array([.25, 0, 0]), 3), reference = shared.clone();
  const a = new T.BufferGeometry().setAttribute('position', shared), b = new T.BufferGeometry().setAttribute('position', shared);
  const first = createGpuBufferGeometry(device, a, {instanceHistory: history});
  const second = createGpuBufferGeometry(device, b, {instanceHistory: history}); gl.update(reference, 0);
  const replacement = new T.BufferAttribute(new Float32Array([.75, 0, 0]), 3), referenceReplacement = replacement.clone();
  a.setAttribute('position', replacement); first.update(); gl.update(referenceReplacement, 0);
  shared.array[0] = reference.array[0] = 9;
  a.dispose(); gl.remove(referenceReplacement); second.update(); gl.update(reference, 0);
  assert.deepEqual(new Uint8Array(second.vertexBuffer.data), gl.get(reference).buffer.bytes);
  assert.equal(new Float32Array(second.vertexBuffer.data)[0], .25);
  first.dispose(); second.dispose(); b.dispose(); history.dispose();
});
