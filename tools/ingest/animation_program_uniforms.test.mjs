// Whole program/geometry owners with recording GPU resources; no shader execution.
import test from 'node:test';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {createProgramUniformArena} from './animation_program_uniforms.mjs';
const root = process.env.F3D_UNIFORM_TEST_ROOT ? pathToFileURL(process.env.F3D_UNIFORM_TEST_ROOT + '/') : new URL('./', import.meta.url);
const {createProgramMeshes} = await import(new URL('animation_program_mesh.mjs', root));
const {createGpuProgramGeometry} = await import(new URL('gpu_buffer_geometry.mjs', root));
const fail = (code, message) => { throw Object.assign(Error(message), {code}); };
function fixture(t, {maxDraws = 32, maxBytes = 1 << 22, maxBufferSize = 1 << 22, alignment = 256, maxBinding = 65536} = {}) {
  const buffers = [], groups = [], pipelines = [], writes = [], clean = [], scopes = [];
  let scopeGate = Promise.resolve(), pipelineGate = Promise.resolve(), lose, scopeError = null, allocationError = null, groupError = null;
  const device = {limits: {maxBufferSize, maxUniformBufferBindingSize: maxBinding, minUniformBufferOffsetAlignment: alignment},
    features: new Set(), lost: new Promise(resolve => { lose = resolve; }),
    pushErrorScope(kind) { scopes.push(kind); }, popErrorScope() { scopes.pop(); return Promise.resolve(null); },
    createBuffer(d) {
      if (allocationError) throw allocationError;
      assert.ok(d.size <= maxBufferSize); assert.equal(d.size % 4, 0);
      const b = {...d, bytes: new Uint8Array(d.size), dead: false, destroy() { this.dead = true; },
        getMappedRange() { return this.bytes.buffer; }, unmap() {}}; buffers.push(b); return b;
    },
    createBindGroupLayout: d => d, createPipelineLayout: d => d, createShaderModule: d => d,
    createBindGroup(d) {
      if (groupError) throw groupError;
      for (const entry of d.entries) if (entry.resource.buffer) {
        assert.ok(entry.resource.size <= maxBinding); assert.ok(entry.resource.size <= entry.resource.buffer.size);
        assert.equal(entry.resource.buffer.dead, false);
      }
      groups.push(d); return d;
    },
    async createRenderPipelineAsync(d) { pipelines.push(d); await pipelineGate; return d; },
    queue: {onSubmittedWorkDone: async () => {}, writeBuffer(buffer, offset, data, start = 0, length) {
      assert.equal(buffer.dead, false); const w = data.BYTES_PER_ELEMENT ?? 1;
      const bytes = new Uint8Array(data.buffer ?? data, (data.byteOffset ?? 0) + start * w,
        length === undefined ? data.byteLength - start * w : length * w);
      assert.equal(bytes.byteLength % 4, 0); buffer.bytes.set(bytes, offset);
      writes.push({buffer, offset, bytes: bytes.slice()});
    }},
  };
  function scoped(_device, fn) {
    device.pushErrorScope('validation'); device.pushErrorScope('out-of-memory');
    let value, error; try { value = fn(); } catch (e) { error = e; }
    device.popErrorScope(); device.popErrorScope();
    const reported = scopeError;
    const errors = scopeGate.then(() => { if (error) throw error; if (reported) throw reported; });
    return {value, error, errors};
  }
  const gpu = createGpuProgramGeometry(device, {attributes: {}, drawRange: {start: 0, count: 3}}, []);
  const owner = createProgramMeshes({device, format: 'rgba32float', depthFormat: null, sampleCount: 1, maxDraws,
    label: 'uniform-test', fail, scoped, lost: device.lost, maxUniformBytes: maxBytes});
  const raster = {blend: null, frontFace: 'ccw', cullMode: 'none', writeMask: 15};
  const program = size => ({vertex: '// compiled vertex fixture', fragment: '// compiled fragment fixture',
    reflection: {textures: [], outputs: [{location: 0}], uniformBufferSize: size}});
  const add = (size, options = {}) => owner.add(gpu, {program: program(size), raster, ...options});
  const stage = (record, value, input = {}) => {
    const bytes = new Uint8Array(record.reflection.uniformBufferSize).fill(value), command = {};
    owner.stage(record, {programUniforms: bytes, ...input}, command); return command;
  };
  t.after(() => { for (const close of clean) close(); owner.dispose(); gpu.dispose(); assert.ok(buffers.every(b => b.dead)); assert.equal(scopes.length, 0); });
  return {device, owner, gpu, add, stage, program, raster, buffers, groups, pipelines, writes, clean, lose, scoped,
    arena(options = {}) { const a = createProgramUniformArena(device, {maxDraws, label: 'uniform-test', fail, scoped, maxBytes, ...options}); clean.push(() => a.dispose()); return a; },
    holdScopes() { let release; scopeGate = new Promise(r => { release = r; }); return release; },
    holdPipelines() { let release; pipelineGate = new Promise(r => { release = r; }); return release; },
    errors({scope = null, allocation = null, group = null} = {}) { scopeError = scope; allocationError = allocation; groupError = group; }};
}
function contents(command) {
  const {group, offset} = command.program, {buffer, size} = group.entries[0].resource;
  assert.equal(group.layout, command.pipeline.layout.bindGroupLayouts[0]);
  assert.ok(offset + size <= buffer.size);
  return buffer.bytes.slice(offset, offset + size);
}
for (const size of [4096, 32768, 65536]) {
  test(`${size}-byte shaders support every configured draw, not a guessed 1 KiB average`, async t => {
    const f = fixture(t), r = await f.add(size), commands = [];
    const allocations = [f.buffers.length, f.groups.length];
    f.owner.begin(); for (let i = 0; i < 32; i++) commands.push(f.stage(r, i + 1));
    f.owner.write();
    assert.deepEqual([f.buffers.length, f.groups.length], allocations, 'submission allocates neither buffers nor groups');
    commands.forEach((c, i) => assert.deepEqual(contents(c), new Uint8Array(size).fill(i + 1)));
    assert.ok(f.owner.uniformDiagnostics.allocatedBytes <= 1 << 22);
  });
}
test('per-buffer limits split capacity into native pages, never one oversized allocation', async t => {
  const f = fixture(t, {maxDraws: 8, maxBufferSize: 65536}), r = await f.add(32768), commands = [];
  f.owner.begin(); for (let i = 0; i < 8; i++) commands.push(f.stage(r, i)); f.owner.write();
  assert.equal(new Set(commands.map(c => c.program.group.entries[0].resource.buffer)).size, 4);
  commands.forEach((c, i) => assert.equal(contents(c)[0], i));
  assert.equal(f.owner.uniformDiagnostics.pages, 4);
});
for (const alignment of [32, 64, 128, 256]) {
  test(`uses supplied ${alignment}-byte alignment and non-overlapping mixed-size packets`, async t => {
    const f = fixture(t, {alignment, maxDraws: 8}), small = await f.add(20), large = await f.add(32768);
    const commands = Array.from({length: 8}, (_, i) => f.stage(i % 2 ? large : small, i + 10)); f.owner.write();
    for (const [i, c] of commands.entries()) {
      assert.equal(c.program.offset % alignment, 0); assert.ok(contents(c).every(x => x === i + 10));
    }
  });
}
test('concurrent programs of the same reflected size share validated layouts and page bindings', async t => {
  const f = fixture(t, {maxDraws: 8}); const records = await Promise.all(Array.from({length: 12}, () => f.add(32768)));
  const commands = records.slice(0, 8).map((r, i) => f.stage(r, i)); f.owner.write();
  assert.equal(new Set(records.map(r => r.pipelines.normal.layout.bindGroupLayouts[0])).size, 1);
  assert.equal(new Set(commands.map(c => c.program.group)).size, 1);
  assert.equal(f.groups.length, 1); assert.equal(f.owner.uniformDiagnostics.bindings, 1);
});
test('growth preserves already staged bytes, original bindings, and every earlier command', async t => {
  const f = fixture(t, {maxDraws: 8, maxBufferSize: 65536}), small = await f.add(16);
  const first = f.stage(small, 1), original = first.program.group;
  const large = await f.add(65536); const second = f.stage(large, 2); f.owner.write();
  assert.equal(first.program.group, original); assert.ok(contents(first).every(x => x === 1)); assert.ok(contents(second).every(x => x === 2));
  assert.equal(original.entries[0].resource.buffer.dead, false);
  f.owner.begin(); const reused = f.stage(small, 3); f.owner.write();
  assert.equal(reused.program.group, original); assert.equal(reused.program.offset, first.program.offset);
});
test('all registered block layouts are bound on newly added pages', async t => {
  const f = fixture(t, {maxDraws: 8, maxBufferSize: 65536}), small = await f.add(32768), large = await f.add(65536);
  const commands = [];
  for (let i = 0; i < 4; i++) commands.push(f.stage(large, i + 1));
  for (let i = 0; i < 4; i++) commands.push(f.stage(small, i + 5));
  f.owner.write(); commands.forEach((c, i) => assert.ok(contents(c).every(x => x === i + 1)));
});
test('tight byte limits fail explicitly without losing draw slots or prior packet contents', async t => {
  const f = fixture(t, {maxDraws: 8, maxBytes: 65536}), small = await f.add(16), large = await f.add(32768);
  const a = f.stage(large, 1); assert.throws(() => f.stage(large, 2, {programUniforms: new Uint8Array(8)}), {code: 'ANIMATION_RENDER_VALUE'});
  const b = f.stage(large, 2);
  assert.throws(() => f.stage(small, 3), {code: 'ANIMATION_RENDER_LIMIT'});
  f.owner.write(); assert.equal(contents(a)[0], 1); assert.equal(contents(b)[0], 2);
  assert.equal(f.owner.uniformDiagnostics.draws, 2);
  assert.equal(f.owner.uniformDiagnostics.allocatedBytes, 65536);
  f.owner.begin(); const c = f.stage(small, 4); f.owner.write(); assert.equal(contents(c)[0], 4);
});
test('draw count is independent of the available bytes and resets at begin', async t => {
  const f = fixture(t, {maxDraws: 2}), r = await f.add(16); f.stage(r, 1); f.stage(r, 2);
  assert.throws(() => f.stage(r, 3), {code: 'ANIMATION_RENDER_LIMIT'});
  f.owner.begin(); f.stage(r, 4); assert.equal(f.owner.uniformDiagnostics.draws, 1);
});
test('subarray snapshots preserve raw float bits after mutation and detachment', async t => {
  const f = fixture(t), r = await f.add(16);
  const data = new Uint32Array([123, 0x7f800001, 0xffc12345, 0x80000000, 0x7f800000, 456]);
  const expected = new Uint8Array(data.buffer, 4, 16).slice(), command = {};
  f.owner.stage(r, {programUniforms: new Uint8Array(data.buffer, 4, 16)}, command);
  data.fill(0); structuredClone(data.buffer, {transfer: [data.buffer]}); f.owner.write();
  assert.deepEqual(contents(command), expected);
  assert.throws(() => f.stage(r, 0, {programUniforms: new Uint8Array(new SharedArrayBuffer(16))}), {code: 'ANIMATION_RENDER_VALUE'});
  assert.equal(f.owner.uniformDiagnostics.draws, 1);
});
test('registration snapshots shader, texture and raster inputs before uniform growth yields', async t => {
  const f = fixture(t), release = f.holdScopes();
  const program = f.program(16), view = {}, sampler = {}, texture = {view, sampler, sampleType: 'float'};
  program.reflection.textures = [{sampleType: 'float', dimension: '2d', textureBinding: 0, samplerBinding: 1}];
  const raster = {...f.raster, blendConstant: [0.1,0.2,0.3,0.4]};
  const pending = f.owner.add(f.gpu, {program, raster, textures: [texture]});
  program.vertex = 'changed'; program.reflection.uniformBufferSize = 65536; program.reflection.textures[0].dimension = 'cube';
  raster.writeMask = 0; raster.blendConstant[0] = 0.9; texture.view = {};
  release(); const r = await pending;
  assert.equal(r.reflection.uniformBufferSize, 16); assert.equal(r.pipelines.normal.vertex.module.code, '// compiled vertex fixture');
  assert.equal(r.pipelines.normal.fragment.targets[0].writeMask, 15); assert.equal(r.blendConstant[0], 0.1);
  assert.equal(r.textureGroup.entries[0].resource, view); assert.equal(r.textureGroup.layout.entries[0].texture.viewDimension, '2d');
});
for (const failure of ['scope', 'allocation', 'group']) {
  test(`${failure} failure rolls back unpublished growth and leaves existing programs drawable`, async t => {
    const f = fixture(t, {maxDraws: 8, maxBufferSize: 65536}), r = await f.add(16), before = f.owner.uniformDiagnostics.allocatedBytes;
    const error = Error(failure); f.errors({[failure]: error}); await assert.rejects(f.add(65536), e => e === error);
    assert.equal(f.owner.uniformDiagnostics.allocatedBytes, before);
    f.errors(); const c = f.stage(r, 9); f.owner.write(); assert.equal(contents(c)[0], 9);
    const retry = await f.add(65536); f.owner.begin(); f.stage(retry, 10); f.owner.write();
  });
}
for (const during of ['registration', 'pipeline']) {
  test(`disposal interrupts ${during} and cannot publish or resurrect native resources`, async t => {
    const f = fixture(t), release = during === 'registration' ? f.holdScopes() : f.holdPipelines();
    const pending = f.add(32768); const rejected = assert.rejects(pending, {code: 'ANIMATION_RENDER_DISPOSED'});
    for (let i = 0; i < 12; i++) await Promise.resolve();
    f.owner.dispose(); await rejected; release();
    assert.equal(f.owner.uniformDiagnostics.allocatedBytes, 0);
    await assert.rejects(f.add(16), {code: 'ANIMATION_RENDER_DISPOSED'});
  });
}
test('device loss interrupts registration, drains unpublished buffers and preserves its error', async t => {
  const f = fixture(t), release = f.holdScopes(); const pending = f.add(32768);
  const rejected = assert.rejects(pending, {code: 'ANIMATION_RENDER_LOST'});
  await Promise.resolve(); f.lose({message: 'fixture loss'}); await rejected; release();
  assert.equal(f.owner.uniformDiagnostics.allocatedBytes, 0);
});
test('invalid reflected sizes never allocate or consume a valid registration', async t => {
  const f = fixture(t); const before = f.buffers.length;
  for (const size of [0, 2, 15, 17, NaN, Infinity, -16, 65540]) await assert.rejects(f.add(size), {code: 'ANIMATION_RENDER_LIMIT'});
  assert.equal(f.buffers.length, before); assert.equal(f.groups.length, 0); await f.add(16);
});
test('arena capacities validate before allocation; foreign binding tokens are refused', async t => {
  const f = fixture(t), before = f.buffers.length;
  for (const patch of [{maxDraws: 0}, {maxDraws: 65537}, {maxBytes: 3}, {maxBytes: NaN}])
    assert.throws(() => f.arena(patch), {code: 'ANIMATION_RENDER_LIMIT'});
  assert.equal(f.buffers.length, before);
  const a = f.arena(), b = f.arena(); const binding = await a.register(16);
  assert.throws(() => b.stage(binding, new Uint8Array(16)), {code: 'ANIMATION_RENDER_VALUE'});
});
test('empty frames write nothing and a smaller next frame uploads only its used prefix', async t => {
  const f = fixture(t, {maxDraws: 8, maxBufferSize: 65536}), r = await f.add(65536), s = await f.add(16);
  for (let i = 0; i < 8; i++) f.stage(r, i); f.owner.write(); assert.equal(f.writes.length, 8);
  f.owner.begin(); f.owner.write(); assert.equal(f.writes.length, 8);
  f.stage(s, 99); f.owner.write(); assert.equal(f.writes.length, 9); assert.equal(f.writes.at(-1).bytes.length, 16);
});

test('scalar uniform blocks retain their native four-byte binding size', async t => {
  const f = fixture(t, {maxDraws: 2}), r = await f.add(4);
  const a = f.stage(r, 1), b = f.stage(r, 2); f.owner.write();
  assert.equal(contents(a).length, 4); assert.equal(contents(b)[0], 2);
});
test('device loss interrupts an outstanding pipeline with the loss error, not an invalid result', async t => {
  const f = fixture(t), release = f.holdPipelines(), pending = f.add(32768);
  const rejected = assert.rejects(pending, {code: 'ANIMATION_RENDER_LOST'});
  for (let i = 0; i < 12; i++) await Promise.resolve();
  f.lose({message: 'pipeline loss'}); await rejected; release();
});
