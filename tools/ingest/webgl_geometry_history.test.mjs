import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.module.js';
import {WebGLAttributes} from '../../upstream/three.js/src/renderers/webgl/WebGLAttributes.js';
import {WebGLGeometries} from '../../upstream/three.js/src/renderers/webgl/WebGLGeometries.js';
import {createWebGLInstanceHistory} from './webgl_instance_history.mjs';
import {createThreePassInstances} from './three_pass_instances.mjs';
import {createThreePassGeometry} from './three_pass_geometry.mjs';
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

function wireframeOracle(geometry) {
  const manager = WebGLGeometries({}, {remove() {}}, {memory: {geometries: 0}}, {releaseStatesOfGeometry() {}});
  return () => { manager.get(null, geometry); return manager.getWireframeAttribute(geometry); };
}

test('queued edge history matches pinned version triggers across private geometry replacements and disposal', () => {
  const owner = createThreePassInstances(T), scene = new T.Scene(), camera = new T.PerspectiveCamera();
  const geometry = new T.BufferGeometry().setAttribute('position', new T.BufferAttribute(new Float32Array(18), 3));
  geometry.attributes.position.version = 5;
  const reference = geometry.clone(); reference.attributes.position.version = 5;
  const expected = wireframeOracle(reference), mesh = new T.Mesh(geometry, new T.MeshBasicMaterial({wireframe: true}));
  mesh.frustumCulled = false; scene.add(mesh);
  const compare = () => {
    const frame = captureThreeMaterialState(T, scene, {camera, selectionOnly: true, allowInstances: true});
    const packet = owner.capture(scene, camera, frame); packet.install();
    const actual = owner.history.wireframeGeometry(packet.geometryView(geometry)).index, result = expected();
    assert.deepEqual(actual.array, result.array); packet.release(); return [...actual.array];
  };
  const initial = compare();
  for (const g of [geometry, reference]) g.setIndex([5, 4, 3]);
  assert.deepEqual(compare(), initial, 'index version zero cannot replace cached nonindexed version five');
  for (const g of [geometry, reference]) g.index.version = 5;
  assert.deepEqual(compare(), initial);
  for (const g of [geometry, reference]) g.index.version = 6;
  const indexed = compare(); assert.deepEqual(indexed, [5, 4, 4, 3, 3, 5]);
  for (const g of [geometry, reference]) g.setIndex([0, 1, 2]);
  assert.deepEqual(compare(), indexed, 'same/lower-version identity replacement retains edges');
  for (const g of [geometry, reference]) { g.setIndex(null); g.attributes.position.needsUpdate = true; }
  assert.deepEqual(compare(), indexed, 'index removal and position updates retain cached edges');
  geometry.dispose(); reference.dispose(); assert.deepEqual(compare(), initial, 'public disposal resets the edge cache');
  owner.dispose(); geometry.dispose(); reference.dispose(); mesh.material.dispose();
});

test('wireframe index width follows the pinned position-count threshold only at regeneration', () => {
  const history = createWebGLInstanceHistory({geometry: true}), captures = createThreePassGeometry(T, history);
  const geometry = new T.BufferGeometry().setAttribute('position', new T.BufferAttribute(new Float32Array(65535 * 3), 3));
  geometry.setIndex([0, 1, 2]);
  const reference = geometry.clone(), expected = wireframeOracle(reference);
  const compare = () => {
    const actual = captures.wireframeGeometry(geometry).index.array, result = expected().array;
    assert.equal(actual.constructor, result.constructor); assert.deepEqual(actual, result); return actual.constructor;
  };
  assert.equal(compare(), Uint32Array);
  geometry.attributes.position.count = reference.attributes.position.count = 65534;
  assert.equal(compare(), Uint32Array);
  geometry.index.needsUpdate = true; reference.index.needsUpdate = true;
  assert.equal(compare(), Uint16Array);
  captures.dispose(); history.dispose(); geometry.dispose(); reference.dispose();
});

test('wireframe packet and preparation budgets include edges without acknowledging the unused original index', () => {
  const geometry = new T.BufferGeometry().setAttribute('position', new T.BufferAttribute(new Float32Array(9), 3)).setIndex([0, 1, 2]);
  const mesh = new T.Mesh(geometry, new T.MeshBasicMaterial({wireframe: true})), scene = new T.Scene(), camera = new T.PerspectiveCamera();
  mesh.frustumCulled = false; scene.add(mesh); geometry.index.addUpdateRange(0, 1); geometry.index.needsUpdate = true;
  const frame = () => captureThreeMaterialState(T, scene, {camera, selectionOnly: true, allowInstances: true});
  const small = createThreePassInstances(T, {maxBytes: 47});
  assert.throws(() => small.capture(scene, camera, frame()), {code: 'F3D_RENDERER_UNSUPPORTED'});
  assert.equal(small.history.bytes, 0); assert.equal(small.pendingBytes, 0); small.dispose();
  const exact = createThreePassInstances(T, {maxBytes: 48}), packet = exact.capture(scene, camera, frame());
  assert.equal(packet.bytes, 48); assert.equal(exact.history.bytes, 36);
  assert.deepEqual(geometry.index.updateRanges, [{start: 0, count: 1}]); packet.release(); exact.dispose();
  const history = createWebGLInstanceHistory({geometry: true}), captures = createThreePassGeometry(T, history, {maxBytes: 16});
  const another = geometry.clone(); captures.wireframeGeometry(geometry, {accept: false});
  assert.throws(() => captures.wireframeGeometry(another, {accept: false}), {code: 'GEOMETRY_GPU_HISTORY'});
  geometry.dispose(); captures.wireframeGeometry(another, {accept: false}); captures.wireframeGeometry(another);
  assert.throws(() => captures.wireframeGeometry(geometry), {code: 'GEOMETRY_GPU_HISTORY'});
  captures.dispose(); history.dispose(); geometry.dispose(); another.dispose(); mesh.material.dispose();
});

test('submitting disposed queued wireframes does not resurrect source identities', () => {
  const history = createWebGLInstanceHistory({geometry: true}), captures = createThreePassGeometry(T, history, {maxGeometries: 1});
  for (let i = 0; i < 8; i++) {
    const geometry = new T.BufferGeometry().setAttribute('position', new T.BufferAttribute(new Float32Array(9), 3)).setIndex([0, 1, 2]);
    const shape = captures.inspect(geometry, 128); shape.edgePlan = captures.edgePlan(geometry);
    captures.validate(new Map([[geometry, shape]]));
    const packet = captures.capture(shape, new Map([[geometry.attributes.position, history.capture(geometry.attributes.position)]]));
    geometry.dispose(); packet.install();
    assert.deepEqual([...captures.wireframeGeometry(packet.view).index.array], [0, 1, 1, 2, 2, 0]);
    packet.release();
    assert.equal(geometry._listeners.dispose.length, 0, 'submitting a retired generation cannot restore its public listener');
  }
  captures.dispose(); history.dispose();
});

test('wireframe size changes retire only private index residency after the submitted draw boundary', () => {
  const device = geometryDevice(), history = createWebGLInstanceHistory({geometry: true}), captures = createThreePassGeometry(T, history);
  const geometry = new T.PlaneGeometry(); geometry.attributes.position.array[0] = .25;
  const wireframe = captures.wireframeGeometry(geometry);
  const gpu = createGpuBufferGeometry(device, wireframe, {maxAttributes: 4, maxBytes: 4096, instanceHistory: history});
  geometry.attributes.position.array[0] = 9;
  for (let i = 1; i <= 140; i++) {
    const previous = bufferGeometrySnapshot(gpu, device), finish = history.beginPrivateSubmission();
    const indices = Array.from({length: i * 3}, (_, j) => j % 3);
    geometry.setIndex(indices); geometry.index.version = i;
    captures.wireframeGeometry(geometry); gpu.update();
    const next = bufferGeometrySnapshot(gpu, device);
    assert.equal(next.indexCount, i * 6);
    assert.equal(gpu.diagnostics.residentAttributes, 4);
    assert.equal(previous.indexBuffer.destroyed, false, 'earlier encoded draws retain their edge buffer until submission');
    assert.equal(new Float32Array(next.vertexBuffers[0].data)[0], .25, 'edge retirement must preserve source vertex upload history');
    assert.equal(next.vertexBuffers[0], previous.vertexBuffers[0]);
    finish();
    assert.equal(previous.indexBuffer.destroyed, true);
    assert.equal(gpu.bufferBytes, 128 + Math.max(4, Math.ceil(indices.length * 4 / 4) * 4));
  }
  assert.equal(device.buffers.filter(buffer => !buffer.destroyed).length, 4);
  gpu.dispose(); captures.dispose(); history.dispose(); geometry.dispose();
  assert.equal(device.buffers.filter(buffer => !buffer.destroyed).length, 0);
});

test('failed private wireframe uploads release both current and deferred index residency', () => {
  const device = geometryDevice(), history = createWebGLInstanceHistory({geometry: true}), captures = createThreePassGeometry(T, history);
  const geometry = new T.PlaneGeometry(), wireframe = captures.wireframeGeometry(geometry);
  const gpu = createGpuBufferGeometry(device, wireframe, {instanceHistory: history});
  const previous = bufferGeometrySnapshot(gpu, device).indexBuffer, finish = history.beginPrivateSubmission();
  geometry.setIndex([0, 1, 2]); geometry.index.version = 1; captures.wireframeGeometry(geometry);
  device.writeError = new Error('failed private index upload');
  assert.throws(() => gpu.update(), /failed private index upload/);
  assert.equal(gpu.bufferBytes, 0); assert.equal(previous.destroyed, true);
  assert.equal(device.buffers.filter(buffer => !buffer.destroyed).length, 0);
  finish(); finish(); assert.equal(gpu.bufferBytes, 0, 'closing a failed submission cannot double release its retired bytes');
  gpu.dispose(); captures.dispose(); history.dispose(); geometry.dispose();
});

test('disposing a private wireframe view preserves replaced attributes still shared by a solid geometry', () => {
  const device = geometryDevice(), history = createWebGLInstanceHistory({geometry: true}), captures = createThreePassGeometry(T, history), gl = oracle();
  const shared = new T.BufferAttribute(new Float32Array([.25, 0, 0, 1, 0, 0, 0, 1, 0]), 3), reference = shared.clone();
  const a = new T.BufferGeometry().setAttribute('position', shared), b = new T.BufferGeometry().setAttribute('position', shared);
  const first = createGpuBufferGeometry(device, captures.wireframeGeometry(a), {instanceHistory: history});
  const second = createGpuBufferGeometry(device, b, {instanceHistory: history}); gl.update(reference, 0);
  a.setAttribute('position', new T.BufferAttribute(new Float32Array(9), 3));
  shared.array[0] = reference.array[0] = 9;
  a.dispose(); second.update(); gl.update(reference, 0);
  assert.deepEqual(new Uint8Array(second.vertexBuffer.data), gl.get(reference).buffer.bytes);
  assert.equal(new Float32Array(second.vertexBuffer.data)[0], .25);
  first.dispose(); second.dispose(); captures.dispose(); history.dispose(); b.dispose();
});
