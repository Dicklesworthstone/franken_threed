// Production geometry, program and index-cache modules; recording GPU, no pixels.
import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuProgramGeometry, programGeometrySnapshot} from './gpu_buffer_geometry.mjs';
import {createProgramMeshes} from './animation_program_mesh.mjs';
import {createProgramIndexLowering} from './animation_program_indices.mjs';
function fixture(t, {C = Uint16Array, data = [0,1,2,65535,3,4,5], maxIndexBytes = 65536} = {}) {
  const buffers = [], pipelines = [], clean = []; let fence = Promise.resolve();
  const fail = (code, message) => { throw Object.assign(Error(message), {code}); };
  const device = {features: new Set(), limits: {maxBufferSize: 1 << 20, maxUniformBufferBindingSize: 65536},
    lost: new Promise(() => {}), pushErrorScope() {}, popErrorScope: async () => null,
    createBuffer(d) { const b = {...d, bytes: new Uint8Array(d.size), destroy() { this.dead = true; },
      getMappedRange() { return this.bytes.buffer; }, unmap() {}}; buffers.push(b); return b; },
    queue: {onSubmittedWorkDone() { return fence; }, writeBuffer(b, offset, data, start = 0, size) {
      const w = data.BYTES_PER_ELEMENT ?? 1;
      b.bytes.set(new Uint8Array(data.buffer ?? data, (data.byteOffset ?? 0) + start * w,
        size === undefined ? data.byteLength - start * w : size * w), offset);
    }}, createBindGroup: d => d, createBindGroupLayout: d => d, createPipelineLayout: d => d, createShaderModule: d => d,
    async createRenderPipelineAsync(d) { pipelines.push(d); return d; }};
  const source = {attributes: {}, drawRange: {start: 0, count: data.length},
    index: {array: new C(data), itemSize: 1, count: data.length, version: 0, normalized: false, updateRanges: [],
      onUploadCallback() {}, clearUpdateRanges() { this.updateRanges.length = 0; }}};
  const gpu = createGpuProgramGeometry(device, source, []);
  const owner = createProgramMeshes({device, format: 'rgba32float', depthFormat: null, sampleCount: 1, maxDraws: 8,
    label: 'restart', maxIndexBytes, fail, lost: device.lost, scoped(_d, fn) { return {value: fn(), errors: Promise.resolve()}; }});
  const add = (topology, flat = false) => owner.add(gpu, {program: {vertex: '// fixture', fragment: '// fixture',
    reflection: {uniformBufferSize: 16, textures: [], outputs: [{location: 0}], varyings: [{flat}]}}, topology,
    stripIndexFormat: C === Uint16Array ? 'uint16' : 'uint32', raster: {blend: null, frontFace: 'ccw', cullMode: 'none', writeMask: 15}});
  const stage = (r, options = {}) => { const c = {}; owner.stage(r, {programUniforms: new Uint8Array(16), ...options}, c); return c; };
  t.after(() => { for (const f of clean) f(); owner.dispose(); gpu.dispose(); assert.ok(buffers.every(b => b.dead)); });
  return {device, source, gpu, owner, add, stage, buffers, pipelines, clean, fail,
    hold() { let release; fence = new Promise(r => { release = r; }); return release; }};
}
function values(c) {
  if (!c.indexBuffer) return [];
  const C = c.indexFormat === 'uint16' ? Uint16Array : Uint32Array;
  return [...new C(c.indexBuffer.bytes.buffer)].slice(c.first, c.first + c.count);
}
for (const C of [Uint8Array, Uint16Array, Uint32Array]) for (const topology of ['triangles', 'lines', 'points']) {
  test(`${C.name}/${topology}: smooth indexed lists split at fixed restart markers`, async t => {
    const restart = 2 ** (8 * C.BYTES_PER_ELEMENT) - 1;
    const f = fixture(t, {C, data: [restart,0,1,2,restart,3,4,5,6,restart]}), r = await f.add(topology);
    const c = f.stage(r);
    assert.deepEqual(values(c), topology === 'triangles' ? [0,1,2,3,4,5] : topology === 'lines' ? [0,1,3,4,5,6] : [0,1,2,3,4,5,6]);
    assert.equal(c.indexFormat, 'uint32'); assert.equal(c.first, 0);
  });
}
for (const topology of ['triangles', 'lines', 'points']) {
  test(`${topology}: no-restart path retains the native buffer even above lowering budget`, async t => {
    const f = fixture(t, {data: [0,1,2,3,4,5], maxIndexBytes: 4}), r = await f.add(topology), before = f.buffers.length;
    const c = f.stage(r, {first: 1, count: 4});
    assert.equal(c.indexBuffer, programGeometrySnapshot(f.gpu, f.device).indexBuffer);
    assert.equal(c.indexFormat, 'uint16'); assert.equal(c.first, 1); assert.equal(c.count, 4);
    assert.equal(f.buffers.length, before);
  });
}
for (const topology of ['triangles', 'lines', 'line-strip']) {
  test(`${topology}: smooth and flat cached schedules cannot alias`, async t => {
    const f = fixture(t), smooth = await f.add(topology), flat = await f.add(topology, true);
    const a = f.stage(smooth), b = f.stage(flat);
    assert.deepEqual(values(a), topology === 'line-strip' ? [0,1,2,65535,3,4,5] : [0,1,2,3,4,5].filter((_v, i) => topology !== 'lines' || i !== 2 && i !== 5));
    assert.deepEqual(values(b), topology === 'triangles' ? [2,0,1,5,3,4] : topology === 'lines' ? [1,0,4,3] : [1,0,2,1,4,3,5,4]);
    assert.notEqual(a.indexBuffer, b.indexBuffer);
  });
}
test('restart grouping begins at the selected range, with no cross-marker primitive', async t => {
  const f = fixture(t, {data: [9,0,1,2,65535,3,4,5,6]}), r = await f.add('triangles');
  f.source.drawRange = {start: 2, count: 6};
  assert.deepEqual(values(f.stage(r)), [3,4,5]);
});
test('empty and all-marker lists draw zero vertices rather than marker indices', async t => {
  const f = fixture(t, {data: [65535,65535]}), r = await f.add('points');
  const c = f.stage(r); assert.equal(c.count, 0); assert.equal(c.indexBuffer, null);
  assert.equal(f.stage(r, {count: 0}).count, 0);
});
test('restart detection uses uploads, including partial writes and callback failures', async t => {
  const f = fixture(t, {data: [0,1,2,3,4,5]}), r = await f.add('lines'), native = f.stage(r).indexBuffer;
  f.source.index.array[2] = 65535;
  assert.equal(f.stage(r).indexBuffer, native);
  f.source.index.version++; f.source.index.updateRanges.push({start: 2, count: 1});
  f.source.index.array[4] = 65535; // This unselected edit must remain invisible.
  f.source.index.onUploadCallback = () => { throw Error('uploaded'); };
  assert.throws(() => f.gpu.update(), /uploaded/);
  assert.deepEqual(values(f.stage(r)), [0,1,3,4]);
});
test('released index arrays still support restart conversion', async t => {
  const f = fixture(t), r = await f.add('triangles'); f.source.index.array = null; f.gpu.update();
  assert.deepEqual(values(f.stage(r)), [0,1,2,3,4,5]);
});
test('negative restart detection is cached by resident revision without allocating buffers', t => {
  const f = fixture(t, {data: [0,1,2,3]}), cache = createProgramIndexLowering(f.device, {label: 'negative', fail: f.fail, maxBytes: 4});
  f.clean.push(() => cache.dispose());
  let reads = 0; const s = programGeometrySnapshot(f.gpu, f.device);
  const g = {...s, readIndex(i) { reads++; return s.readIndex(i); }};
  const before = f.buffers.length;
  assert.equal(cache.prepare(f.gpu, g, 'points', 0, 4, false), null); assert.equal(reads, 4);
  for (let i = 0; i < 8; i++) { cache.begin(); assert.equal(cache.prepare(f.gpu, g, 'points', 0, 4, false), null); }
  assert.equal(reads, 4); assert.equal(f.buffers.length, before); assert.equal(cache.allocatedBytes, 0);
});
test('a new no-marker revision returns to the native buffer and fences old converted storage', async t => {
  const f = fixture(t), r = await f.add('triangles'); f.owner.begin(); const a = f.stage(r), release = f.hold();
  f.source.index.array.set([0,1,2,3,4,5,6]); f.source.index.version++; f.gpu.update(); f.owner.begin(); const b = f.stage(r);
  assert.equal(b.indexBuffer, programGeometrySnapshot(f.gpu, f.device).indexBuffer);
  assert.equal(a.indexBuffer.dead, undefined); release(); await Promise.resolve(); await Promise.resolve();
  assert.equal(a.indexBuffer.dead, true);
});
test('smooth restart lowering enforces its native allocation limit before creating a buffer', async t => {
  const f = fixture(t, {maxIndexBytes: 4}), r = await f.add('triangles'), before = f.buffers.length;
  assert.throws(() => f.stage(r), {code: 'ANIMATION_RENDER_LIMIT'}); assert.equal(f.buffers.length, before);
});
