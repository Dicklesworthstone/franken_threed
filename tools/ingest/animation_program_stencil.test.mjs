// Real program/geometry owners and draw encoders with a recording GPUDevice.
// These are native-command and ownership tests, not shader/pixel execution.
import test from 'node:test';
import assert from 'node:assert/strict';
import {createProgramMeshes} from './animation_program_mesh.mjs';
import {createGpuProgramGeometry} from './gpu_buffer_geometry.mjs';
import {createAnimationRenderBundleCache, encodeAnimationRenderSpans} from './animation_render_bundles.mjs';

const FACE = {compare: 'equal', failOp: 'replace', depthFailOp: 'increment-wrap', passOp: 'decrement-clamp'};
function raster(reference = 7) {
  return {blend: null, writeMask: 15, cullMode: 'none', frontFace: 'ccw',
    depthWriteEnabled: false, depthCompare: 'less-equal',
    stencil: {front: {...FACE}, back: {...FACE}, readMask: 0x3f, writeMask: 0xf0}, stencilReference: reference};
}
function fixture(t, {depthFormat = 'depth24plus-stencil8', format = 'rgba8unorm', sampleCount = 1, indexed = false, points = false} = {}) {
  const descriptors = [], buffers = [], writes = [], scopes = [];
  let pipelineGate = Promise.resolve(), pipelineFailure = null;
  const device = {features: new Set(['depth32float-stencil8']), limits: {maxBufferSize: 1 << 20, maxUniformBufferBindingSize: 65536},
    lost: new Promise(() => {}), pushErrorScope() {}, popErrorScope: async () => null,
    queue: {writeBuffer(buffer, offset, data, start = 0, length = data.byteLength) {
      const bytes = new Uint8Array(data.buffer ?? data, (data.byteOffset ?? 0) + start, length);
      new Uint8Array(buffer.bytes).set(bytes, offset); writes.push({buffer, offset, bytes: bytes.slice()});
    }, onSubmittedWorkDone: async () => {}},
    createBuffer(d) { const b = {bytes: new ArrayBuffer(d.size), getMappedRange() { return this.bytes; }, unmap() {}, destroy() { this.dead = true; }}; buffers.push(b); return b; },
    createBindGroupLayout: d => d, createBindGroup: d => d, createPipelineLayout: d => d, createShaderModule: d => d,
    async createRenderPipelineAsync(d) { descriptors.push(d); await pipelineGate; if (pipelineFailure) throw pipelineFailure; return d; },
    createRenderBundleEncoder(d) {
      assert.equal(d.stencilReadOnly, false);
      const draws = []; let pipeline;
      return {setPipeline(p) { pipeline = p; }, setBindGroup() {}, setVertexBuffer() {}, setIndexBuffer() {},
        draw(...args) { draws.push({pipeline, args, indexed: false}); }, drawIndexed(...args) { draws.push({pipeline, args, indexed: true}); },
        finish() { return {draws}; }};
    }};
  const index = indexed ? {array: new Uint16Array([0, 1, 2]), itemSize: 1, count: 3, version: 0, onUploadCallback() {}} : null;
  const gpu = createGpuProgramGeometry(device, {attributes: {}, index, drawRange: {start: 0, count: 3}}, []);
  const program = {vertex: '// fixture', fragment: '// fixture', reflection: {points, textures: [], outputs: [{location: 0}], uniformBufferSize: 16}};
  const owner = createProgramMeshes({device, format, depthFormat, sampleCount, maxDraws: 8, label: 'program-stencil',
    fail(code, message) { throw Object.assign(Error(message), {code}); },
    scoped(_device, operation) { scopes.push(true); return {value: operation(), errors: Promise.resolve()}; }, lost: device.lost});
  const add = (state = raster()) => owner.add(gpu, {program, raster: state, topology: points ? 'points' : 'triangles'});
  const stage = (record, words = [1, 2, 3, 4]) => {
    const command = {}; owner.stage(record, {programUniforms: new Uint8Array(new Float32Array(words).buffer)}, command); return command;
  };
  t.after(() => { owner.dispose(); gpu.dispose(); assert.ok(buffers.every(b => b.dead)); });
  return {device, owner, gpu, program, add, stage, descriptors, buffers, writes, scopes,
    hold() { let resolve; pipelineGate = new Promise(r => { resolve = r; }); return resolve; },
    failPipeline(error) { pipelineFailure = error; }};
}
for (const depthFormat of ['depth24plus-stencil8', 'depth32float-stencil8']) for (const sampleCount of [1, 4]) {
  test(`${depthFormat}/${sampleCount}: both winding pipelines receive stencil operations and masks`, async t => {
    const f = fixture(t, {depthFormat, sampleCount}); const record = await f.add();
    assert.equal(f.descriptors.length, 2);
    for (const d of f.descriptors) {
      assert.equal(d.depthStencil.format, depthFormat);
      assert.deepEqual(d.depthStencil.stencilFront, FACE); assert.deepEqual(d.depthStencil.stencilBack, FACE);
      assert.equal(d.depthStencil.stencilReadMask, 0x3f); assert.equal(d.depthStencil.stencilWriteMask, 0xf0);
      assert.equal(d.depthStencil.depthWriteEnabled, false); assert.equal(d.multisample.count, sampleCount);
    }
    assert.equal(record.pipelines.normal.primitive.frontFace, 'ccw');
    assert.equal(record.pipelines.flipped.primitive.frontFace, 'cw');
    assert.equal(f.stage(record).stencilReference, 7);
  });
}
for (const operation of ['keep', 'zero', 'replace', 'invert', 'increment-clamp', 'decrement-clamp', 'increment-wrap', 'decrement-wrap']) {
  test(`compiled program preserves ${operation} for all stencil outcomes`, async t => {
    const f = fixture(t), r = raster(); r.stencil.front = r.stencil.back = {compare: 'always', failOp: operation, depthFailOp: operation, passOp: operation};
    await f.add(r);
    for (const d of f.descriptors) assert.deepEqual(d.depthStencil.stencilFront, r.stencil.front);
  });
}
for (const depthFormat of [null, 'depth16unorm', 'depth24plus', 'depth32float']) {
  test(`${depthFormat}: a source stencil test has no effect without stencil storage`, async t => {
    const f = fixture(t, {depthFormat}); const record = await f.add();
    for (const d of f.descriptors) assert.equal(d.depthStencil?.stencilFront, undefined);
    assert.equal(f.stage(record).stencilReference, null);
  });
}
test('stencil reference and face state are captured before awaiting pipelines', async t => {
  const f = fixture(t), release = f.hold(), r = raster(13); const pending = f.add(r);
  r.stencilReference = 88; r.stencil.front.compare = 'never'; r.stencil.writeMask = 0;
  release(); const record = await pending;
  assert.equal(f.stage(record).stencilReference, 13);
  assert.equal(record.pipelines.normal.depthStencil.stencilFront.compare, 'equal');
  assert.equal(record.pipelines.normal.depthStencil.stencilWriteMask, 0xf0);
});
test('invalid stencil inputs fail before geometry lookup, scopes, or program allocation', async t => {
  const f = fixture(t); const before = [f.buffers.length, f.scopes.length, f.descriptors.length];
  for (const patch of [{stencilReference: 256}, {stencilReference: -1}, {stencil: {readMask: -1}},
    {stencil: {front: {compare: 'invalid'}}}, {stencil: {back: {passOp: 'invalid'}}}]) {
    await assert.rejects(f.owner.add(null, {program: f.program, raster: {...raster(), ...patch}}), {code: 'ANIMATION_RASTER_INPUT'});
  }
  assert.deepEqual([f.buffers.length, f.scopes.length, f.descriptors.length], before);
  assert.equal((await f.add()).stencilReference, 7);
});
for (const options of [{indexed: true}, {points: true}, {format: null}, {format: 'rgba32float'}]) {
  test(`stencil survives ${JSON.stringify(options)} program draw preparation`, async t => {
    const f = fixture(t, options); const r = await f.add(raster(0)); const command = f.stage(r);
    assert.equal(command.stencilReference, 0);
    assert.equal(command.count, options.points ? 6 : 3);
    if (options.points) assert.equal(command.instanceCount, 3);
    if (options.indexed) assert.ok(command.indexBuffer);
  });
}
for (const bundled of [false, true]) {
  test(`references are applied in source order with ${bundled ? 'cached bundles' : 'direct draws'}`, async t => {
    const f = fixture(t), records = await Promise.all([f.add(raster(9)), f.add(raster(2)), f.add(raster(0))]);
    const cache = bundled ? createAnimationRenderBundleCache(f.device, {format: 'rgba8unorm', depthFormat: 'depth24plus-stencil8', sampleCount: 1, stride: 256, maxDraws: 8}) : null;
    t.after(() => cache?.dispose());
    const trace = []; let reference = -1;
    const pass = {setStencilReference(value) { reference = value; }, setPipeline() {}, setBindGroup() {}, setVertexBuffer() {}, setIndexBuffer() {},
      draw() { trace.push(reference); }, drawIndexed() { trace.push(reference); }, executeBundles(bundles) { for (const b of bundles) for (const _ of b.draws) trace.push(reference); }};
    const parameters = {stride: 256, instancing: false, lightGroup: null};
    for (let frame = 0; frame < 2; frame++) {
      f.owner.begin(); const commands = records.map(r => f.stage(r));
      assert.deepEqual(commands.map(c => c.program.offset), [0, 256, 512]);
      assert.equal(encodeAnimationRenderSpans(pass, commands, commands.length, parameters, cache), 3);
      f.owner.write();
    }
    assert.deepEqual(trace, [9, 2, 0, 9, 2, 0]);
    if (bundled) { assert.equal(cache.diagnostics.builds, 3); assert.equal(cache.diagnostics.reuses, 3); }
  });
}
test('a disabled stencil material has no stencil state and cannot write the attachment', async t => {
  const f = fixture(t), r = raster(); delete r.stencil; delete r.stencilReference;
  const record = await f.add(r);
  assert.equal(record.stencilReference, null);
  for (const d of f.descriptors) assert.equal(d.depthStencil.stencilFront, undefined);
});
test('native pipeline rejection is propagated instead of accepting a stencil mesh', async t => {
  const f = fixture(t), error = Error('native validation failed'); f.failPipeline(error);
  await assert.rejects(f.add(), e => e === error);
});
