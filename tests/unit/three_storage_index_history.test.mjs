/** Source index/storage history at the renderer's capture boundary. The
 * recording device observes submitted bytes; it does not execute shaders. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {withThreeCompute} from '../../tools/ingest/three_compute.mjs';
import {createThreeNodeBundles} from '../../tools/ingest/three_node_bundles.mjs';
import {createGpuProgramGeometry, programGeometrySnapshot} from '../../tools/ingest/gpu_buffer_geometry.mjs';
import {captureComputeStorage} from '../../tools/ingest/gpu_compute_attributes.mjs';
import {geometryDevice} from '../../tools/ingest/fixtures/gpu_geometry_device.mjs';

const Compute = withThreeCompute(class {}, T);
const program = [{name: 'position', location: 0, locations: 1, components: 3, scalar: 'float'}];

function fixture(t) {
  const device = geometryDevice();
  // Exercise the exported compute mixin's storage owner without constructing
  // a renderer canvas or compiling source shaders.
  const renderer = {isWebGPURenderer: true, _device: device, _targetLifetime: new AbortController(),
    _nodes: {nodeFrame: {updateNode() {}}}, _getSampledTextureInputs: () => ({whenIdle: () => Promise.resolve()}),
    info: {memory: {readbackBuffersSize: 0, total: 0}}};
  const bundles = createThreeNodeBundles(T);
  const storage = bundles.resources(Compute.prototype._getStorageInputs.call(renderer));
  const geometry = new T.BufferGeometry().setAttribute('position',
    new T.Float32BufferAttribute([0,0,0,1,0,0,0,1,0,1,1,0], 3));
  const index = new T.StorageBufferAttribute(new Uint32Array([0,1,2,2,3,0]), 1);
  geometry.setIndex(index);
  const dispose = () => { bundles.dispose(); Compute.prototype._disposeResources.call(renderer, run => run()); };
  t.after(dispose);
  function frame() {
    const uses = [], vertices = [], first = new Map();
    return {uses, vertices, capture(renderId) {
      // Source Geometries admits one upload per attribute in this render.
      // Later shader bindings have their own storage.capture boundary.
      return bundles.captureGeometry(geometry, null, renderId, program, (attribute, view) => {
        const previous = first.get(attribute), lease = previous ? storage.retain(previous) : storage.capture(attribute);
        if (!previous) first.set(attribute, lease);
        uses.push(lease); vertices.push(lease);
        return storage.vertexAttribute(lease, view);
      });
    }};
  }
  function submit(frame, packets, shader = null) {
    const history = storage.validate(frame.uses);
    const current = packets.map(packet => packet.refreshStorage(view => {
      for (const lease of frame.vertices) {
        const refreshed = history.refreshVertexAttribute(lease, view);
        if (refreshed !== null) return refreshed;
      }
      return null;
    }) ?? packet);
    const indices = current.map(packet => packet.geometry.index), gpu = [];
    try {
      storage.withReads(frame.uses, () => {
        const encoder = device.createCommandEncoder(), pass = encoder.beginRenderPass({});
        pass.setPipeline({vertex: {entryPoint: 'main', buffers: [{}]}});
        for (const packet of current) {
          const resident = createGpuProgramGeometry(device, packet.geometry, program, {attributeTypes: 'webgpu'});
          gpu.push(resident);
          const snapshot = programGeometrySnapshot(resident, device);
          pass.setVertexBuffer(0, snapshot.vertexBuffers[0]); pass.setIndexBuffer(snapshot.indexBuffer, snapshot.indexFormat);
          pass.setBindGroup(0, {entries: shader ? [{binding: 0, resource: storage.resolve(shader)}] : []});
          pass.drawIndexed(snapshot.indexCount, 1, 0, 0, 0);
        }
        pass.end(); device.queue.submit([encoder.finish()]);
      });
      return indices;
    } finally {
      for (const resident of gpu) resident.dispose();
      for (let i = 0; i < packets.length; i++) {
        if (current[i] !== packets[i]) current[i].release();
        packets[i].release();
      }
      for (const lease of frame.uses) storage.release(lease);
    }
  }
  return {device, storage, index, frame, submit, dispose};
}
const words = (device, buffer) => device.snapshots.at(-1).map(draw => [...new Uint32Array(draw.contents.get(buffer).buffer)]);
const writes = (device, buffer) => device.writes.filter(write => write.buffer === buffer)
  .map(write => [write.offset, write.input.byteLength]);

test('index geometry and later shader uploads preserve ordered partial ranges in the same allocation', t => {
  const f = fixture(t), first = f.frame();
  f.submit(first, [first.capture(1)]);
  const buffer = captureComputeStorage(f.device, f.index).resolve().buffer;
  assert.equal(buffer.usage & (16 | 128), 16 | 128);
  // Initialize as an index before the simulated GPU producer. The pinned
  // source creates INDEX|STORAGE only for this index-first lifecycle.
  new Uint32Array(buffer.data)[5] = 1;
  const frame = f.frame();
  f.index.array.fill(99); f.index.array[0] = 3;
  f.index.addUpdateRange(0, 1); f.index.needsUpdate = true;
  const a = frame.capture(2);
  f.index.array[3] = 1; f.index.addUpdateRange(3, 1); f.index.needsUpdate = true;
  const b = frame.capture(2);
  assert.equal(a.geometry.index, b.geometry.index, 'second geometry use retains the first admitted upload');
  const shader = f.storage.capture(f.index); frame.uses.push(shader);
  f.index.array.fill(99); f.index.needsUpdate = true;
  f.submit(frame, [a, b], shader);
  assert.deepEqual(words(f.device, buffer), [[3,1,2,1,3,1], [3,1,2,1,3,1]]);
  assert.deepEqual(writes(f.device, buffer), [[0,24], [0,4], [12,4]]);
  assert.equal(f.index.version, 3);
  f.dispose(); assert.equal(buffer.destroyed, true);
});

test('retained index views follow admitted uploads while ignoring later CPU requests and retaining warm identity', t => {
  const f = fixture(t), first = f.frame(), packet = first.capture(1);
  const kept = packet.retain(), retained = f.storage.retain(first.vertices[0]), original = packet.geometry.index;
  t.after(() => { kept.release(); f.storage.release(retained); });
  f.submit(first, [packet]);
  const buffer = captureComputeStorage(f.device, f.index).resolve().buffer;
  const warm = () => {
    const lease = f.storage.refreshStorage(retained) ?? f.storage.retain(retained);
    return f.submit({uses: [lease], vertices: [lease]}, [kept.retain()])[0];
  };
  new Uint32Array(buffer.data).set([3,2,1,1,2,0]); f.index.array.fill(99);
  assert.equal(warm(), original, 'same-version GPU writes keep the accepted index view');
  assert.deepEqual(words(f.device, buffer), [[3,2,1,1,2,0]]);
  f.index.array[3] = 3; f.index.addUpdateRange(3, 1); f.index.needsUpdate = true;
  const admitted = f.storage.capture(f.index); t.after(() => f.storage.release(admitted));
  f.index.array.fill(99); f.index.needsUpdate = true;
  const next = warm();
  assert.equal(next.version, 1); assert.notEqual(next, original);
  assert.deepEqual(words(f.device, buffer), [[3,2,1,3,2,0]]);
  assert.equal(warm(), next, 'repeated accepted versions reuse the refreshed geometry proof');
  assert.deepEqual(words(f.device, buffer), [[3,2,1,3,2,0]]);
  assert.deepEqual(writes(f.device, buffer), [[0,24], [12,4]]);
});

for (const dynamic of [false, true])
test(`${dynamic ? 'dynamic' : 'versioned'} first index use can follow an earlier shader-only upload`, t => {
  const f = fixture(t), first = f.frame();
  if (dynamic) f.index.setUsage(T.DynamicDrawUsage);
  f.submit(first, [first.capture(1)]);
  const buffer = captureComputeStorage(f.device, f.index).resolve().buffer;
  new Uint32Array(buffer.data)[5] = 1;
  const frame = f.frame();
  f.index.array.fill(99); f.index.array[0] = 3; f.index.addUpdateRange(0, 1);
  if (!dynamic) f.index.needsUpdate = true;
  const shader = f.storage.capture(f.index); frame.uses.push(shader);
  f.index.array[3] = 1; f.index.addUpdateRange(3, 1);
  if (!dynamic) f.index.needsUpdate = true;
  const packet = frame.capture(2);
  f.index.array.fill(99);
  f.submit(frame, [packet], shader);
  assert.deepEqual(words(f.device, buffer), [[3,1,2,1,3,1]]);
  assert.deepEqual(writes(f.device, buffer), [[0,24], [0,4], [12,4]]);
});

test('an accepted queued index read can precede the next render\'s first geometry upload', t => {
  const f = fixture(t), first = f.frame(), packet = first.capture(1);
  const kept = packet.retain(), retained = f.storage.retain(first.vertices[0]);
  t.after(() => { kept.release(); f.storage.release(retained); });
  // Capture acceptance precedes the FIFO upload. A warm bundle may retain
  // this exact lease before any allocation has reached the recording device.
  f.storage.validate(first.uses);
  assert.equal(f.device.buffers.length, 0);
  const frame = f.frame(), warm = f.storage.retain(retained);
  assert.equal(warm, retained, 'the bundle resource wrapper retains the same opaque lease');
  frame.uses.push(warm); frame.vertices.push(warm);
  f.index.array[0] = 3; f.index.addUpdateRange(0, 1); f.index.needsUpdate = true;
  const ordinary = frame.capture(2);
  assert.doesNotThrow(() => f.storage.validate(frame.uses));
  f.index.array.fill(99); f.index.needsUpdate = true;
  f.submit(first, [packet]);
  const firstDraw = f.device.snapshots[0][0], buffer = firstDraw.index.buffer;
  assert.deepEqual([...new Uint32Array(firstDraw.contents.get(buffer).buffer)], [0,1,2,2,3,0]);
  f.submit(frame, [kept.retain(), ordinary]);
  assert.deepEqual(words(f.device, buffer), [[3,1,2,2,3,0], [3,1,2,2,3,0]]);
  assert.deepEqual(writes(f.device, buffer), [[0,24], [0,4]]);
});

test('rejected duplicate geometry admissions retain their guard and cannot combine indirect histories', t => {
  const f = fixture(t), first = f.storage.capture(f.index);
  f.storage.vertexAttribute(first, Object.create(f.index)).install();
  f.index.array[0] = 3; f.index.needsUpdate = true;
  const second = f.storage.capture(f.index);
  f.storage.vertexAttribute(second, Object.create(f.index)).install();
  const indirect = f.storage.capture(f.index, {indirect: true});
  t.after(() => { for (const lease of [first, second, indirect]) f.storage.release(lease); });
  assert.throws(() => f.storage.validate([first, second]), {code: 'THREE_COMPUTE_READ_CONFLICT'});
  assert.throws(() => f.storage.withReads([first, second], () => assert.fail('rejected histories cannot issue')),
    {code: 'THREE_COMPUTE_READ_CONFLICT'}, 'failed validation must not consume either fresh geometry admission');
  assert.throws(() => f.storage.validate([first, indirect]), {code: 'THREE_COMPUTE_READ_CONFLICT'});
  assert.equal(f.device.writes.length, 0);
});
