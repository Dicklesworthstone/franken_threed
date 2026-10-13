import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuProgramGeometry, programGeometrySnapshot, createGpuBufferGeometry} from './gpu_buffer_geometry.mjs';

// Real residency implementation; this device records uploads, not GPU pixels.
function device(t, maximum = 1 << 20) {
  const buffers = [], writes = [], scopes = [], cleanup = []; let lose, nativeError = null;
  const d = {limits: {maxBufferSize: maximum}, lost: new Promise(r => {lose = r;}),
    pushErrorScope(k) { scopes.push(k); }, popErrorScope: async () => {scopes.pop(); return nativeError;},
    createBuffer(desc) { const b = {...desc, bytes: new Uint8Array(desc.size), dead: false, destroy() {this.dead = true;}}; buffers.push(b); return b; },
    queue: {writeBuffer(b, offset, data, start = 0, size = data.byteLength) {
      assert.equal(b.dead, false); assert.equal(offset % 4, 0); assert.equal(size % 4, 0);
      const bytes = new Uint8Array(data.buffer ?? data, (data.byteOffset ?? 0) + start, size);
      b.bytes.set(bytes, offset); writes.push({b, offset, bytes: bytes.slice()});
    }, onSubmittedWorkDone: async () => {}}};
  t.after(() => {for (const close of cleanup) close(); assert.equal(scopes.length, 0); assert.ok(buffers.every(b => b.dead));});
  return {d, buffers, writes, cleanup, lose, fail(error) {nativeError = error;}};
}
function attr(array, itemSize = 1, divisor = 1, extra = {}) {
  return {array, itemSize, count: array.length / itemSize, isInstancedBufferAttribute: true,
    meshPerAttribute: divisor, version: 0, updateRanges: [], normalized: false, uploads: 0,
    onUploadCallback() {this.uploads++;}, clearUpdateRanges() {this.updateRanges.length = 0;}, ...extra};
}
const reflection = (name, location = 0, scalar = 'float', locations = 1, components = 1) => ({name, location, scalar, locations, components});
function geometry(attributes, extra = {}) {
  const listeners = new Set();
  return {attributes, isInstancedBufferGeometry: true, drawRange: {start: 0, count: 3},
    addEventListener(_k, f) {listeners.add(f);}, removeEventListener(_k, f) {listeners.delete(f);},
    dispose() {for (const f of listeners) f();}, ...extra};
}
function own(t, f, g, refs, options) {
  const h = createGpuProgramGeometry(f.d, g, refs, options); f.cleanup.push(() => h.dispose()); return h;
}
function values(h, f, slot = 0, C = Float32Array) {
  return [...new C(programGeometrySnapshot(h, f.d).vertexBuffers[slot].bytes.buffer)];
}
for (const divisor of [1, 2, 3, 7, 32]) for (const width of [1, 2, 4, 16]) {
  test(`divisor ${divisor} repeats width-${width} records without changing vertex count`, t => {
    const f = device(t), a = attr(Float32Array.from({length: width * 3}, (_, i) => i + 0.25), width, divisor);
    const h = own(t, f, geometry({offset: a}), [reflection('offset', 0, 'float', width === 16 ? 4 : 1, width === 16 ? 4 : width)]);
    const expected = []; for (let row = 0; row < 3; row++) for (let r = 0; r < divisor; r++) expected.push(...a.array.slice(row * width, (row + 1) * width));
    assert.deepEqual(values(h, f), expected);
    const s = programGeometrySnapshot(h, f.d); assert.equal(s.instanceCapacity, 3 * divisor); assert.equal(s.vertexCount, 3);
    assert.equal(s.layouts[0].stepMode, 'instance'); assert.equal(s.layouts[0].arrayStride, width * 4);
    assert.equal(h.bufferBytes, expected.length * 4); assert.equal(h.shadowBytes, h.bufferBytes); assert.equal(a.uploads, 1);
    const before = f.writes.length; h.update(); assert.equal(f.writes.length, before);
  });
}
test('Float32 subarrays preserve NaN payloads, infinities and signed zero in every replica', t => {
  const f = device(t), bits = new Uint32Array([123, 0x7f800001, 0xffc12345, 0x80000000, 0x7f800000, 456]);
  const a = attr(new Float32Array(bits.buffer, 4, 4), 2, 3), h = own(t, f, geometry({a}), [reflection('a', 0, 'float', 1, 2)]);
  assert.deepEqual(values(h, f, 0, Uint32Array), [0x7f800001, 0xffc12345, 0x7f800001, 0xffc12345, 0x7f800001, 0xffc12345,
    0x80000000, 0x7f800000, 0x80000000, 0x7f800000, 0x80000000, 0x7f800000]);
});
for (const [array, extra, C, expected] of [
  [new Uint8Array([0, 255, 128, 64]), {normalized: true}, Float32Array, [0, 1, Math.fround(128 / 255), Math.fround(64 / 255)]],
  [new Int8Array([-128, 127, 0, -64]), {normalized: true}, Float32Array, [-1, 1, 0, Math.fround(-64 / 127)]],
  [new Uint16Array([0x3c00, 0xc000, 0x8000, 0x0001]), {isFloat16BufferAttribute: true}, Float32Array, [1, -2, -0, 2 ** -24]],
  [new Uint32Array([0xffffffff, 0x80000001, 16777217, 7]), {gpuType: 1013}, Uint32Array, [0xffffffff, 0x80000001, 16777217, 7]],
  [new Int16Array([-32768, -1, 32767, 0]), {gpuType: 1013}, Int32Array, [-32768, -1, 32767, 0]],
]) {
  test(`${array.constructor.name}/${C.name}: conversion precedes whole-record replication`, t => {
    const f = device(t), a = attr(array, 2, 2, extra), scalar = C === Uint32Array ? 'uint' : C === Int32Array ? 'int' : 'float';
    const h = own(t, f, geometry({a}), [reflection('a', 0, scalar, 1, 2)]);
    assert.deepEqual(values(h, f, 0, C), [...expected.slice(0, 2), ...expected.slice(0, 2), ...expected.slice(2), ...expected.slice(2)]);
  });
}
test('interleaved attributes share one expanded buffer, upload callback and column layout', t => {
  const f = device(t), data = attr(new Float32Array([1, 2, 10, 11, 3, 4, 20, 21]), 4, 3,
    {isInstancedBufferAttribute: false, isInstancedInterleavedBuffer: true, stride: 4});
  const a = {isInterleavedBufferAttribute: true, data, itemSize: 2, offset: 0, count: 2};
  const b = {...a, offset: 2};
  const h = own(t, f, geometry({a, b}), [reflection('a', 0, 'float', 1, 2), reflection('b', 1, 'float', 1, 2)]);
  assert.equal(f.buffers.length, 1); assert.equal(data.uploads, 1); assert.equal(h.bufferBytes, 96);
  assert.deepEqual(values(h, f), [1,2,10,11,1,2,10,11,1,2,10,11,3,4,20,21,3,4,20,21,3,4,20,21]);
  assert.deepEqual(programGeometrySnapshot(h, f.d).layouts[0].attributes.map(a => a.offset), [0, 8]);
  // A partial source component update must NOT upload unversioned neighbors.
  data.array.set([99, 88, 77, 66]); data.version++; data.updateRanges.push({start: 2, count: 1}); h.update();
  assert.deepEqual(values(h, f).slice(0, 12), [1,2,77,11,1,2,77,11,1,2,77,11]);
  assert.equal(data.updateRanges.length, 0); assert.equal(data.uploads, 2); assert.equal(f.writes.at(-1).bytes.length, 48);
});
test('partial updates crossing source records retain other uploaded elements', t => {
  const f = device(t), a = attr(new Float32Array([1,2,3,4,5,6,7,8]), 2, 3);
  const h = own(t, f, geometry({a}), [reflection('a', 0, 'float', 1, 2)]);
  a.array.fill(100); a.version++; a.updateRanges.push({start: 3, count: 2}); h.update();
  assert.deepEqual(values(h,f), [1,2,1,2,1,2,3,100,3,100,3,100,100,6,100,6,100,6,7,8,7,8,7,8]);
  assert.equal(f.writes.at(-1).offset, 24); assert.equal(f.writes.at(-1).bytes.length, 48);
});
test('pinned zero-length update range copies the rest, first use leaves ranges intact', t => {
  const f = device(t), a = attr(new Float32Array([1,2,3,4]), 2, 2); a.updateRanges.push({start: 1, count: 0});
  const h = own(t, f, geometry({a}), [reflection('a', 0, 'float', 1, 2)]); assert.equal(a.updateRanges.length, 1);
  a.array.set([8,9,10,11]); a.version++; h.update();
  assert.deepEqual(values(h,f), [1,9,1,9,10,11,10,11]); assert.equal(a.updateRanges.length, 0);
});
test('capacity is the minimum count times divisor across independent instance streams', t => {
  const f = device(t), a = attr(new Float32Array([1,2]), 1, 3), b = attr(new Float32Array([3,4,5]), 1, 4);
  const h = own(t,f,geometry({a,b}), [reflection('a'),reflection('b',1)]); assert.equal(programGeometrySnapshot(h,f.d).instanceCapacity,6);
});
test('released source arrays remain resident; release/recreate uses the new divisor', t => {
  const f = device(t), a = attr(new Float32Array([4,5]),1,3), g = geometry({a});
  const h = own(t,f,g,[reflection('a')]); a.array=null; h.update(); assert.deepEqual(values(h,f),[4,4,4,5,5,5]);
  a.array=new Float32Array([6,7]); a.meshPerAttribute=2;
  assert.throws(()=>h.update(),{code:'GEOMETRY_GPU_FORMAT'}); h.release(); h.update(); assert.deepEqual(values(h,f),[6,6,7,7]);
});
for (const budget of ['maxBytes', 'maxInitialBytes', 'maxBufferSize']) {
  test(`${budget} is checked against expanded storage before allocation or callback`, t => {
    const f=device(t,budget==='maxBufferSize'?32:1<<20), a=attr(new Float32Array([1,2]),1,8);
    assert.throws(()=>createGpuProgramGeometry(f.d,geometry({a}),[reflection('a')],budget==='maxBufferSize'?{}:{[budget]:32}),{code:'GEOMETRY_GPU_LIMIT'});
    assert.equal(f.buffers.length,0); assert.equal(f.writes.length,0); assert.equal(a.uploads,0);
  });
}
test('a refused additional expanded attribute leaves prior storage usable',t=>{
  const f=device(t),a=attr(new Float32Array([1]),1,2),g=geometry({a}),h=own(t,f,g,[reflection('a'),reflection('b',1)]);
  g.attributes.b=attr(new Float32Array([2]),1,32);
  const before=f.writes.length;
  assert.throws(()=>h.update({maxAdditionalBytes:64}),{code:'GEOMETRY_GPU_LIMIT'}); assert.equal(f.writes.length,before); assert.deepEqual(values(h,f),[1,1]);
  h.update({maxAdditionalBytes:128}); assert.equal(h.bufferBytes,136);
});
for(const divisor of [0,-1,1.5,NaN,Infinity,0x100000000,'2']) {
  test(`invalid divisor ${String(divisor)} is refused without GPU work`,t=>{
    const f=device(t); assert.throws(()=>createGpuProgramGeometry(f.d,geometry({a:attr(new Float32Array([1]),1,divisor)}),[reflection('a')]),{code:'GEOMETRY_GPU_SHAPE'});
    assert.equal(f.buffers.length,0);
  });
}
test('partial final interleaved records use zero padding without reading outside the view',t=>{
  const f=device(t),data=attr(new Float32Array([1,2,99,3,4]),1,2,{stride:3,isInstancedInterleavedBuffer:true});
  const a={isInterleavedBufferAttribute:true,data,itemSize:2,offset:0,count:2};
  const h=own(t,f,geometry({a}),[reflection('a',0,'float',1,2)]);
  assert.deepEqual(values(h,f),[1,2,99,1,2,99,3,4,0,3,4,0]);
});
test('interleaved final attribute must fit, including its offset',t=>{
  const f=device(t),data=attr(new Float32Array(5),1,2,{stride:3,isInstancedInterleavedBuffer:true});
  assert.throws(()=>createGpuProgramGeometry(f.d,geometry({a:{isInterleavedBufferAttribute:true,data,itemSize:2,offset:1,count:2}}),[reflection('a')]),{code:'GEOMETRY_GPU_SHAPE'});
  assert.equal(f.buffers.length,0);
});
test('source disposal recreates replicas and device loss drains ownership',async t=>{
  const f=device(t),a=attr(new Float32Array([1,2]),1,4),g=geometry({a}),h=own(t,f,g,[reflection('a')]);
  g.dispose();assert.equal(h.bufferBytes,0);h.update();assert.equal(h.bufferBytes,32);
  f.lose({message:'gone'});await assert.rejects(h.whenIdle(),{code:'GEOMETRY_GPU_LOST'});assert.equal(h.bufferBytes,0);
});
test('ordinary vertex/index residency remains byte-exact and does not repeat',t=>{
  const f=device(t),position=attr(new Float32Array([1,2,3,4,5,6,7,8,9]),3,9,{isInstancedBufferAttribute:false});
  const g=geometry({position},{isInstancedBufferGeometry:false});const h=createGpuBufferGeometry(f.d,g);f.cleanup.push(()=>h.dispose());
  assert.deepEqual([...new Float32Array(h.vertexBuffer.bytes.buffer)],[1,2,3,4,5,6,7,8,9]);assert.equal(h.bufferBytes,36);
});
