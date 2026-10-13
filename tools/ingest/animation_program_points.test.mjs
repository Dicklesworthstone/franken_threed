// Whole geometry/program owners and native draw encoders; recording GPU, no pixels.
import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuProgramGeometry, programGeometrySnapshot} from './gpu_buffer_geometry.mjs';
import {createProgramMeshes} from './animation_program_mesh.mjs';
import {createAnimationRenderBundleCache, encodeAnimationRenderSpans} from './animation_render_bundles.mjs';

function attribute(array, itemSize = 1, extra = {}) {
  return {array, itemSize, count: array.length / itemSize, version: 0, normalized: false, updateRanges: [],
    onUploadCallback() {}, clearUpdateRanges() { this.updateRanges.length = 0; }, ...extra};
}
function fixture(t, {data = [4,5,1,1,2,0], C = Uint16Array, vertices = 8, indexed = true, format = 'rgba8unorm',
  depthFormat = 'depth24plus-stencil8', sampleCount = 1, maxPointDraws = 65536, instanced = false, channels = true} = {}) {
  const buffers = [], pipelines = [], uploads = [], encoded = [];
  const fail = (code, message) => { throw Object.assign(Error(message), {code}); };
  function encoder() {
    const calls = [];
    return {setPipeline(p) { assert.ok(p); }, setBindGroup() {}, setVertexBuffer(_slot, b) { assert.equal(b.dead, false); },
      setIndexBuffer() { throw Error('Point quads must not use source indices as quad-corner indices'); },
      draw(...args) { calls.push(args); }, drawIndexed() { throw Error('Unexpected indexed native draw'); }, finish() { return {calls}; }};
  }
  const device = {features: new Set(['depth32float-stencil8']), limits: {maxBufferSize: 1 << 20, maxUniformBufferBindingSize: 65536},
    lost: new Promise(() => {}), pushErrorScope() {}, popErrorScope: async () => null,
    createBuffer(d) { const b = {...d, bytes: new Uint8Array(d.size), dead: false, destroy() { this.dead = true; },
      getMappedRange() { return this.bytes.buffer; }, unmap() {}}; buffers.push(b); return b; },
    queue: {onSubmittedWorkDone: async () => {}, writeBuffer(b, offset, data, start = 0, size) {
      const width = data.BYTES_PER_ELEMENT ?? 1;
      const bytes = new Uint8Array(data.buffer ?? data, (data.byteOffset ?? 0) + start * width,
        size === undefined ? data.byteLength - start * width : size * width);
      b.bytes.set(bytes, offset); uploads.push(bytes.slice());
    }}, createBindGroup: d => d, createBindGroupLayout: d => d, createPipelineLayout: d => d, createShaderModule: d => d,
    async createRenderPipelineAsync(d) { pipelines.push(d); return d; },
    createRenderBundleEncoder() { const e = encoder(); encoded.push(e); return e; }};
  const position = attribute(Float32Array.from({length: vertices * 3}, (_, i) => i / 4), 3);
  const source = {attributes: channels ? {position} : {}, index: indexed ? attribute(new C(data)) : null,
    isInstancedBufferGeometry: instanced, drawRange: {start: 0, count: indexed ? data.length : vertices}};
  const attributes = channels ? [{name: 'position', location: 0, locations: 1, components: 3, scalar: 'float'}] : [];
  const gpu = createGpuProgramGeometry(device, source, attributes);
  const owner = createProgramMeshes({device, format, depthFormat, sampleCount, maxDraws: 8, maxPointDraws, label: 'point-test', fail,
    lost: device.lost, scoped(_d, fn) { return {value: fn(), errors: Promise.resolve()}; }});
  const raster = {blend: null, writeMask: 15, cullMode: 'none', frontFace: 'ccw', depthCompare: 'less-equal', depthWriteEnabled: true,
    ...(depthFormat?.includes('stencil') ? {stencil: {front: {compare: 'equal'}, back: {compare: 'equal'}, writeMask: 0}, stencilReference: 7} : {})};
  const program = {vertex: '// compiled point-shader fixture', fragment: '// compiled fragment fixture',
    reflection: {points: true, uniformBufferSize: 16, textures: [], outputs: [{location: 0}], varyings: []}};
  const add = () => owner.add(gpu, {program, raster, topology: 'points'});
  const stage = (record, patch = {}) => { const c = {}; owner.stage(record, {programUniforms: new Uint8Array(16), ...patch}, c); return c; };
  const cleanup = [];
  t.after(() => { for (const fn of cleanup) fn(); owner.dispose(); gpu.dispose(); assert.ok(buffers.every(b => b.dead)); });
  function render(commands, cache = null) {
    const events = []; let reference = null;
    const pass = {...encoder(), setStencilReference(v) { reference = v; },
      draw(...args) { events.push({args, reference}); },
      executeBundles(bundles) { for (const b of bundles) for (const args of b.calls) events.push({args, reference}); }};
    const count = encodeAnimationRenderSpans(pass, commands, commands.length, {stride: 256, instancing: true}, cache);
    return {count, events};
  }
  const cache = () => { const c = createAnimationRenderBundleCache(device, {format, depthFormat, sampleCount, maxDraws: 8, stride: 256}); cleanup.push(() => c.dispose()); return c; };
  return {device, gpu, source, owner, add, stage, render, cache, buffers, pipelines, uploads, encoded};
}
const ids = ({events}) => events.flatMap(({args: [, count, , first]}) => Array.from({length: count}, (_, i) => first + i));
for (const C of [Uint16Array, Uint32Array]) for (const bundled of [false, true]) {
  test(`${C.name}/${bundled ? 'bundled' : 'direct'}: indexed particles preserve IDs, duplicates and order`, async t => {
    const f = fixture(t, {C}), r = await f.add(), c = f.stage(r), before = [f.buffers.length, f.uploads.length];
    assert.deepEqual(c.program.pointRuns, [4,2,1,1,1,2,0,1]); assert.equal(c.pointCount, 6);
    assert.ok(Object.isFrozen(c.program.pointRuns)); assert.equal(c.indexBuffer, null);
    const rendered = f.render([c], bundled ? f.cache() : null);
    assert.equal(rendered.count, 4); assert.deepEqual(ids(rendered), [4,5,1,1,2,0]);
    assert.ok(rendered.events.every(e => e.reference === 7 && e.args[0] === 6 && e.args[2] === 0));
    assert.deepEqual([f.buffers.length, f.uploads.length], before);
    for (const d of f.pipelines) assert.equal(d.vertex.buffers[0].stepMode, 'instance');
    assert.equal(c.vertexBuffers[0], programGeometrySnapshot(f.gpu, f.device).vertexBuffers[0]);
  });
}
for (const C of [Uint16Array, Uint32Array]) {
  const restart = C === Uint16Array ? 65535 : 4294967295;
  test(`${C.name}: fixed-index restart values never become point IDs`, async t => {
    const f = fixture(t, {C, data: [restart,0,1,restart,2,3,restart]}), r = await f.add();
    const c = f.stage(r); assert.deepEqual(c.program.pointRuns, [0,2,2,2]);
    assert.equal(c.pointCount, 4); assert.deepEqual(ids(f.render([c])), [0,1,2,3]);
  });
}
for (const patch of [{count: 0}, {first: 99}, {instanceCount: 0}]) {
  test(`empty point selection ${JSON.stringify(patch)} encodes no native draw`, async t => {
    const f = fixture(t), r = await f.add(), c = f.stage(r, patch);
    assert.equal(c.pointCount, 0); assert.deepEqual(c.program.pointRuns, []);
    assert.equal(f.render([c]).count, 0); assert.equal(f.render([c], f.cache()).count, 0);
  });
}
test('all-restart selection emits no phantom particle', async t => {
  const f = fixture(t, {data: [65535,65535]}), r = await f.add();
  assert.equal(f.render([f.stage(r)]).count, 0);
});
test('selected draw/group range intersects source geometry range before planning', async t => {
  const f = fixture(t), r = await f.add(); f.source.drawRange = {start: 2, count: 3};
  assert.deepEqual(ids(f.render([f.stage(r, {first: 1, count: 3})])), [1,1]);
});
test('nonindexed sprites retain one native draw and their source start ID', async t => {
  const f = fixture(t, {indexed: false}), r = await f.add();
  const rendered = f.render([f.stage(r, {first: 2, count: 3})]);
  assert.equal(rendered.count, 1); assert.deepEqual(ids(rendered), [2,3,4]);
});
test('first planning and subsequent uses read uploaded index history, not CPU edits', async t => {
  const f = fixture(t), r = await f.add(); f.source.index.array.fill(7);
  assert.deepEqual(ids(f.render([f.stage(r)])), [4,5,1,1,2,0]);
  f.source.index.version++; f.source.index.updateRanges.push({start: 1, count: 1}); f.gpu.update();
  assert.deepEqual(ids(f.render([f.stage(r)])), [4,7,1,1,2,0]);
});
test('a throwing upload callback still exposes successfully submitted index bytes', async t => {
  const f = fixture(t), r = await f.add();
  f.source.index.array[0] = 6; f.source.index.version++;
  f.source.index.onUploadCallback = () => { throw Error('after upload'); };
  assert.throws(() => f.gpu.update(), /after upload/);
  assert.deepEqual(ids(f.render([f.stage(r)])), [6,5,1,1,2,0]);
});
test('released source index arrays do not prevent point rendering', async t => {
  const f = fixture(t), r = await f.add(); f.source.index.array = null; f.gpu.update();
  assert.deepEqual(ids(f.render([f.stage(r)])), [4,5,1,1,2,0]);
});
test('immutable earlier plans survive same-frame index uploads', async t => {
  const f = fixture(t), r = await f.add(), a = f.stage(r);
  f.source.index.array.fill(3); f.source.index.version++; f.gpu.update(); const b = f.stage(r);
  assert.deepEqual(ids(f.render([a,b])), [4,5,1,1,2,0,3,3,3,3,3,3]);
});
test('bundle reuse compares the full point order, not only first/count/resource IDs', async t => {
  const f = fixture(t), r = await f.add(), cache = f.cache();
  f.owner.begin(); f.render([f.stage(r)], cache);
  f.owner.begin(); f.render([f.stage(r)], cache);
  assert.equal(cache.diagnostics.builds, 1); assert.equal(cache.diagnostics.reuses, 1);
  f.source.index.array.set([1,2,4,4,5,0]); f.source.index.version++; f.gpu.update();
  f.owner.begin(); assert.deepEqual(ids(f.render([f.stage(r)], cache)), [1,2,4,4,5,0]);
  assert.equal(cache.diagnostics.builds, 2);
});
test('one shared expansion budget bounds all point draws before staging', async t => {
  const f = fixture(t, {maxPointDraws: 4}), r = await f.add(); f.owner.begin();
  assert.equal(f.stage(r).program.offset, 0);
  assert.throws(() => f.stage(r, {first: 0, count: 1}), {code: 'ANIMATION_RENDER_LIMIT'});
  f.owner.begin(); assert.equal(f.stage(r, {first: 0, count: 1}).program.offset, 0);
  assert.throws(() => f.stage(r), {code: 'ANIMATION_RENDER_LIMIT'});
  const c = f.stage(r, {first: 1, count: 1}); assert.equal(c.program.offset, 256);
});
for (const patch of [{first: -1}, {first: NaN}, {first: 0.5}, {count: -1}, {count: Infinity}, {instanceCount: 2}]) {
  test(`invalid point input ${Object.keys(patch)[0]}=${Object.values(patch)[0]} leaves the owner reusable`, async t => {
    const f = fixture(t), r = await f.add(), before = f.buffers.length;
    assert.throws(() => f.stage(r, patch), {code: 'ANIMATION_RENDER_GEOMETRY'});
    assert.equal(f.stage(r).program.offset, 0); assert.equal(f.buffers.length, before);
  });
}
test('invalid resident vertex index is rejected before native draw encoding', async t => {
  const f = fixture(t, {data: [1,8]}), r = await f.add();
  assert.throws(() => f.stage(r), {code: 'ANIMATION_RENDER_GEOMETRY'});
  assert.deepEqual(ids(f.render([f.stage(r, {count: 1})])), [1]);
});
test('attribute-free point shaders retain large unsigned source vertex IDs', async t => {
  const f = fixture(t, {C: Uint32Array, data: [4000000000,4000000001], channels: false}), r = await f.add();
  assert.deepEqual(f.stage(r).program.pointRuns, [4000000000,2]);
});
for (const config of [{format: 'rgba32float'}, {format: null}, {format: 'rgba16float', sampleCount: 4}, {depthFormat: null}]) {
  test(`indexed point planning retains attachment state ${JSON.stringify(config)}`, async t => {
    const f = fixture(t, config), r = await f.add();
    assert.deepEqual(ids(f.render([f.stage(r)])), [4,5,1,1,2,0]);
    for (const p of f.pipelines) assert.equal(p.multisample.count, config.sampleCount ?? 1);
  });
}
test('source-instanced point rejection remains explicit, before program allocation', async t => {
  const f = fixture(t, {instanced: true}), before = f.buffers.length;
  await assert.rejects(f.add(), {code: 'ANIMATION_RENDER_GEOMETRY'}); assert.equal(f.buffers.length, before);
});
