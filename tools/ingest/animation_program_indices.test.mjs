// Whole geometry/program modules, recording GPU resources; no shader execution.
import test from 'node:test';
import assert from 'node:assert/strict';
const root = process.env.F3D_INDEX_TEST_ROOT ? new URL(`file://${process.env.F3D_INDEX_TEST_ROOT}/`) : new URL('./', import.meta.url);
const {createGpuProgramGeometry, programGeometrySnapshot, createGpuBufferGeometry} = await import(new URL('gpu_buffer_geometry.mjs', root));
const {createProgramMeshes} = await import(new URL('animation_program_mesh.mjs', root));
import {createProgramIndexLowering} from './animation_program_indices.mjs';
const raster = {blend: null, writeMask: 15, frontFace: 'ccw', cullMode: 'none', depthCompare: 'less-equal', depthWriteEnabled: true};
function attribute(array, extra = {}) {
  return {array, itemSize: 1, count: array.length, version: 0, updateRanges: [], normalized: false,
    onUploadCallback() {}, clearUpdateRanges() { this.updateRanges.length = 0; }, ...extra};
}
function fixture(t, {data = [0,1,2,3,4,5], C = Uint16Array, indexed = true, maxBytes = 1 << 20, maxEntries = 16} = {}) {
  const buffers = [], descriptors = [], cleanup = []; let fence = Promise.resolve(), failMapping = false;
  const device = {features: new Set(), limits: {maxBufferSize: 1 << 20, maxUniformBufferBindingSize: 65536}, lost: new Promise(() => {}),
    pushErrorScope() {}, popErrorScope: async () => null,
    queue: {onSubmittedWorkDone() { return fence; }, writeBuffer(b, offset, data, start = 0, size) {
      const width = data.BYTES_PER_ELEMENT ?? 1;
      b.bytes.set(new Uint8Array(data.buffer ?? data, (data.byteOffset ?? 0) + start * width, size === undefined ? data.byteLength - start * width : size * width), offset);
    }},
    createBuffer(d) { const b = {...d, bytes: new Uint8Array(d.size), dead: false, destroy() { this.dead = true; },
      getMappedRange() { if (failMapping) throw Error('mapping failed'); return this.bytes.buffer; }, unmap() {}}; buffers.push(b); return b; },
    createBindGroupLayout: d => d, createBindGroup: d => d, createPipelineLayout: d => d, createShaderModule: d => d,
    async createRenderPipelineAsync(d) { descriptors.push(d); return d; }};
  const source = {attributes: {}, index: indexed ? attribute(new C(data)) : null, drawRange: {start: 0, count: data.length}};
  const gpu = createGpuProgramGeometry(device, source, []);
  const fail = (code, message) => { throw Object.assign(Error(message), {code}); };
  const owner = createProgramMeshes({device, format: 'rgba32float', depthFormat: 'depth24plus', sampleCount: 1,
    maxDraws: maxEntries, label: 'flat-test', fail, lost: device.lost, maxIndexBytes: maxBytes,
    scoped(_d, fn) { return {value: fn(), errors: Promise.resolve()}; }});
  t.after(() => { for (const close of cleanup) close(); owner.dispose(); gpu.dispose(); assert.ok(buffers.every(b => b.dead)); });
  const add = (topology, flat = true) => owner.add(gpu, {program: {vertex: '// fixture', fragment: '// fixture',
    reflection: {uniformBufferSize: 16, textures: [], outputs: [{location: 0}], varyings: [{flat}]}}, raster, topology,
    ...(indexed && topology === 'line-strip' ? {stripIndexFormat: C === Uint16Array ? 'uint16' : 'uint32'} : {})});
  const stage = (r, options = {}) => { const c = {}; owner.stage(r, {programUniforms: new Uint8Array(16), ...options}, c); return c; };
  return {device, source, gpu, owner, add, stage, buffers, descriptors, cleanup,
    hold() { let release; fence = new Promise(r => { release = r; }); return release; },
    breakMapping() { failMapping = true; }};
}
const words = c => c.indexBuffer ? [...new Uint32Array(c.indexBuffer.bytes.buffer)].slice(c.first, c.first + c.count) : [];
for (const indexed of [false, true]) for (const C of [Uint8Array, Uint16Array, Uint32Array]) {
  for (const [topology, expected] of [['triangles', [2,0,1,5,3,4]], ['lines', [1,0,3,2,5,4]], ['line-strip', [1,0,2,1,3,2,4,3,5,4]]]) {
    test(`${topology}/${C.name}/${indexed}: preserves primitive order and last-vertex values`, async t => {
      const f = fixture(t, {indexed, C}); const r = await f.add(topology); f.owner.begin(); const c = f.stage(r);
      assert.deepEqual(words(c), expected); assert.equal(c.first, 0); assert.equal(c.indexFormat, 'uint32');
      assert.equal(c.pipeline.primitive.topology, topology === 'triangles' ? 'triangle-list' : 'line-list');
      assert.equal(c.pipeline.primitive.stripIndexFormat, undefined);
    });
  }
}
for (const C of [Uint8Array, Uint16Array, Uint32Array]) {
  const restart = 2 ** (8 * C.BYTES_PER_ELEMENT) - 1;
  for (const [topology, expected] of [['triangles', [2,0,1,6,4,5]], ['lines', [1,0,5,4,7,6]], ['line-strip', [1,0,2,1,5,4,6,5,7,6]]]) {
    test(`${C.name}/${topology}: restart markers do not become vertices or bridge strips`, async t => {
      const f = fixture(t, {C, data: [restart,0,1,2,restart,restart,4,5,6,7,restart,9]}); const r = await f.add(topology);
      assert.deepEqual(words(f.stage(r)), expected);
    });
  }
}
for (const topology of ['triangles', 'lines', 'line-strip']) {
  test(`${topology}: draw and geometry ranges bound grouping before lowering`, async t => {
    const f = fixture(t, {data: [9,8,7,6,5,4,3,2]}); const r = await f.add(topology);
    f.source.drawRange = {start: 2, count: 5};
    const c = f.stage(r, {first: 1, count: 5}); // indices 2..5, not an old full-buffer phase
    assert.deepEqual(words(c), topology === 'triangles' ? [5,7,6] : topology === 'lines' ? [6,7,4,5] : [6,7,5,6,4,5]);
  });
}
test('first lowering observes resident bytes, not unversioned CPU edits or replacement arrays', async t => {
  const f = fixture(t); const r = await f.add('triangles');
  f.source.index.array.fill(99); f.source.index.array = new Uint16Array([8,8,8,8,8,8]);
  assert.deepEqual(words(f.stage(r)), [2,0,1,5,3,4]);
});
test('partial uploads do not expose unselected edits; source-only versions do not invalidate', async t => {
  const f = fixture(t); const r = await f.add('triangles'); const first = f.stage(r);
  f.source.index.array.fill(9); f.source.index.version++;
  assert.equal(f.stage(r).indexBuffer, first.indexBuffer);
  f.source.index.updateRanges.push({start: 1, count: 1}); f.gpu.update();
  const second = f.stage(r); assert.deepEqual(words(second), [2,0,9,5,3,4]);
  assert.notEqual(second.indexBuffer, first.indexBuffer); assert.equal(first.indexBuffer.dead, false);
});
test('successful uploads followed by a callback failure still advance index history', async t => {
  const f = fixture(t); const r = await f.add('triangles'); const first = f.stage(r);
  f.source.index.array[2] = 12; f.source.index.version++; f.source.index.updateRanges.push({start: 2, count: 1});
  f.source.index.onUploadCallback = () => { throw Error('callback failed'); };
  assert.throws(() => f.gpu.update(), /callback failed/);
  assert.deepEqual(words(f.stage(r)), [12,0,1,5,3,4]); assert.equal(first.indexBuffer.dead, false);
});
test('released CPU index arrays remain usable for new lowered ranges and no-op updates', async t => {
  const f = fixture(t); const r = await f.add('line-strip');
  f.source.index.array = null; f.gpu.update();
  assert.deepEqual(words(f.stage(r, {first: 2, count: 3})), [3,2,4,3]);
});
test('ordinary index residency also survives an onUpload array release', t => {
  const f = fixture(t); const index = attribute(new Uint16Array([0,1,2]));
  index.onUploadCallback = () => { index.array = null; };
  const position = attribute(new Float32Array(9), {itemSize: 3, count: 3});
  const g = createGpuBufferGeometry(f.device, {attributes: {position}, index}); f.cleanup.push(() => g.dispose());
  assert.doesNotThrow(() => g.update()); assert.equal(g.bufferBytes, 44);
});
test('mixed line and triangle programs never collide in the index cache', async t => {
  const f = fixture(t); const [tri, line] = await Promise.all([f.add('triangles'), f.add('lines')]);
  f.owner.begin(); const a = f.stage(tri), b = f.stage(line);
  assert.deepEqual(words(a), [2,0,1,5,3,4]); assert.deepEqual(words(b), [1,0,3,2,5,4]);
  assert.notEqual(a.indexBuffer, b.indexBuffer); assert.equal(a.indexBuffer.dead, false);
});
test('replaced index storage invalidates lowering without changing program layout', async t => {
  const f = fixture(t); const r = await f.add('triangles'); const a = f.stage(r);
  f.source.index = attribute(new Uint32Array([6,7,8,9,10,11])); f.gpu.update();
  assert.deepEqual(words(f.stage(r)), [8,6,7,11,9,10]); assert.equal(a.indexBuffer.dead, false);
});
test('superseded same-frame buffers retire only after the next frame fence', async t => {
  const f = fixture(t); const r = await f.add('triangles'); f.owner.begin(); const a = f.stage(r);
  f.source.index.array[0] = 7; f.source.index.version++; f.gpu.update(); const b = f.stage(r);
  assert.equal(a.indexBuffer.dead, false); const release = f.hold(); f.owner.begin();
  await Promise.resolve(); assert.equal(a.indexBuffer.dead, false);
  release(); await Promise.resolve(); await Promise.resolve(); assert.equal(a.indexBuffer.dead, true);
  assert.equal(b.indexBuffer.dead, false); assert.equal(f.stage(r).indexBuffer, b.indexBuffer);
});
test('retired buffers remain charged until their completion fence settles', async t => {
  const f = fixture(t, {maxBytes: 48}); const r = await f.add('triangles'); f.owner.begin(); f.stage(r);
  f.source.index.array[0] = 7; f.source.index.version++; f.gpu.update(); f.stage(r);
  f.source.index.array[0] = 8; f.source.index.version++; f.gpu.update();
  assert.throws(() => f.stage(r), {code: 'ANIMATION_RENDER_LIMIT'});
  const release = f.hold(); f.owner.begin(); assert.throws(() => f.stage(r), {code: 'ANIMATION_RENDER_LIMIT'});
  release(); await Promise.resolve(); await Promise.resolve(); assert.deepEqual(words(f.stage(r)), [2,8,1,5,3,4]);
});
test('empty strips and incomplete lists allocate no lowered buffer', async t => {
  const f = fixture(t); const r = await f.add('line-strip'); const before = f.buffers.length;
  for (const count of [0,1]) assert.equal(f.stage(r, {count}).count, 0);
  assert.equal(f.buffers.length, before);
});
test('unchanged topology is reused across frames; smooth strips retain native strip pipelines', async t => {
  const f = fixture(t); const flat = await f.add('triangles'), smooth = await f.add('line-strip', false);
  f.owner.begin(); const a = f.stage(flat); f.owner.begin(); assert.equal(f.stage(flat).indexBuffer, a.indexBuffer);
  const b = f.stage(smooth); assert.equal(b.pipeline.primitive.topology, 'line-strip');
  assert.equal(b.pipeline.primitive.stripIndexFormat, 'uint16'); assert.equal(b.indexFormat, 'uint16');
});
test('index allocation and mapping failures release the new resource', async t => {
  const f = fixture(t); const r = await f.add('triangles'); f.breakMapping();
  assert.throws(() => f.stage(r), /mapping failed/); assert.equal(f.buffers.at(-1).dead, true);
});
test('entry bounds and size limits reject before allocating unbounded buffers', t => {
  const f = fixture(t); const cache = createProgramIndexLowering(f.device, {label: 'bounds', fail: (code, message) => { throw Object.assign(Error(message), {code}); }, maxBytes: 64, maxEntries: 1});
  f.cleanup.push(() => cache.dispose()); const s = programGeometrySnapshot(f.gpu, f.device); cache.begin();
  cache.prepare(f.gpu, s, 'triangles', 0, 3); const before = f.buffers.length;
  assert.throws(() => cache.prepare(f.gpu, s, 'lines', 0, 4), {code: 'ANIMATION_RENDER_LIMIT'});
  assert.equal(f.buffers.length, before);
  assert.throws(() => cache.prepare({}, {indexBuffer: null, vertexCount: 1e9}, 'line-strip', 0, 1e9), {code: 'ANIMATION_RENDER_LIMIT'});
});
