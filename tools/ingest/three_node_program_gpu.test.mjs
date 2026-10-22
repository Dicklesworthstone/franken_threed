// Recording GPU tests check ownership, native binding layout and byte history.
// Actual WGSL execution is compared with pinned r186 in the browser parity lane.
import test from 'node:test';
import assert from 'node:assert/strict';
import {createThreeNodeProgramMeshes} from './three_node_program_gpu.mjs';
import {createGpuProgramGeometry} from './gpu_buffer_geometry.mjs';
const fail = (code, message) => { throw Object.assign(Error(message), {code}); };
function rig(t, {maxDraws = 8, maxUniformBytes = 4 * 1024 * 1024, maxBindGroups = 4, maxBufferSize = 1 << 22} = {}) {
  const buffers = [], bindGroups = [], pipelines = [], writes = [];
  let resolveLost, releasePipelines = null, pipelineGate = Promise.resolve(), pipelineError = null, scopeError = null;
  const device = {features: new Set(), limits: {maxBufferSize, maxUniformBufferBindingSize: 65536,
    minUniformBufferOffsetAlignment: 256, maxBindGroups: 4}, lost: new Promise(resolve => { resolveLost = resolve; }),
    pushErrorScope() {}, popErrorScope: async () => null,
    createBuffer(descriptor) {
      assert.ok(descriptor.size <= maxBufferSize);
      const buffer = {...descriptor, bytes: new Uint8Array(descriptor.size), dead: false,
        getMappedRange() { return this.bytes.buffer; }, unmap() {}, destroy() { this.dead = true; }};
      buffers.push(buffer); return buffer;
    },
    createBindGroupLayout: d => d, createPipelineLayout: d => d, createShaderModule: d => d,
    createBindGroup(descriptor) { bindGroups.push(descriptor); return descriptor; },
    async createRenderPipelineAsync(descriptor) {
      pipelines.push(descriptor); await pipelineGate; if (pipelineError) throw pipelineError; return descriptor;
    },
    queue: {onSubmittedWorkDone: async () => {}, writeBuffer(buffer, offset, data, start = 0, count) {
      assert.equal(buffer.dead, false); const size = data.BYTES_PER_ELEMENT ?? 1;
      const bytes = new Uint8Array(data.buffer ?? data, (data.byteOffset ?? 0) + start * size,
        count === undefined ? data.byteLength - start * size : count * size);
      buffer.bytes.set(bytes, offset); writes.push({buffer, offset, bytes: bytes.slice()});
    }},
  };
  const scoped = (_device, fn) => {
    let value, error; try { value = fn(); } catch (e) { error = e; }
    const errors = error || scopeError ? Promise.reject(error ?? scopeError) : Promise.resolve(); errors.catch(() => {});
    return {value, error, errors};
  };
  const source = {attributes: {position: {array: new Float32Array([-.5, -.5, 0, .5, -.5, 0, 0, .5, 0]),
    itemSize: 3, count: 3, version: 0, normalized: false, updateRanges: [], clearUpdateRanges() {}, onUploadCallback() {}}},
    index: {array: new Uint16Array([0, 1, 2]), itemSize: 1, count: 3, version: 0, normalized: false,
      updateRanges: [], clearUpdateRanges() {}, onUploadCallback() {}}, drawRange: {start: 0, count: 3}};
  const attributes = [{name: 'position', location: 0, locations: 1, components: 3, scalar: 'float'}];
  const gpu = createGpuProgramGeometry(device, source, attributes);
  const owner = createThreeNodeProgramMeshes({device, format: 'rgba8unorm', depthFormat: 'depth24plus',
    sampleCount: 1, maxDraws, label: 'native-node-test', fail, scoped, maxUniformBytes, maxBindGroups});
  const raster = {blend: null, frontFace: 'ccw', cullMode: 'back', writeMask: 15,
    depthWriteEnabled: true, depthCompare: 'less-equal'};
  const program = (sizes = [128, 96]) => ({nodeProgram: true, vertex: '// native vertex', fragment: '// native fragment',
    groups: sizes.map((byteLength, index) => ({index, name: `group-${index}`, byteLength}))});
  const add = (sizes, options = {}) => owner.add(gpu, {nodeProgram: program(sizes), raster, ...options});
  const stage = (record, values = [1, 2], input = {}) => {
    const nodeUniforms = record.groups.map((group, i) => new Uint8Array(group.byteLength).fill(values[i] ?? values[0]));
    const command = {}; owner.stage(record, {nodeUniforms, ...input}, command); return command;
  };
  t.after(() => { releasePipelines?.(); owner.dispose(); gpu.dispose(); assert.ok(buffers.every(b => b.dead)); });
  return {device, source, gpu, owner, raster, program, add, stage, buffers, bindGroups, pipelines, writes,
    lose: resolveLost,
    holdPipelines() { pipelineGate = new Promise(resolve => { releasePipelines = resolve; }); return releasePipelines; },
    errors({pipeline = null, scope = null} = {}) { pipelineError = pipeline; scopeError = scope; }};
}
function packet(command, index) {
  const {group, offset} = command.nodeProgram.groups[index];
  const {buffer, size} = group.entries[0].resource;
  assert.equal(group.layout, command.pipeline.layout.bindGroupLayouts[index]);
  return buffer.bytes.slice(offset, offset + size);
}
test('native graphics uses main entrypoints, native group order and F3D-owned pipelines', async t => {
  const f = rig(t), record = await f.add();
  assert.equal(record.nodeProgram, true); assert.equal(record.program, undefined);
  assert.equal(record.pipelines.normal.vertex.entryPoint, 'main');
  assert.equal(record.pipelines.normal.fragment.entryPoint, 'main');
  assert.equal(record.pipelines.normal.primitive.frontFace, 'ccw');
  assert.equal(record.pipelines.flipped.primitive.frontFace, 'cw');
  assert.equal(record.pipelines.normal.primitive.cullMode, 'back');
  assert.equal(record.pipelines.normal.depthStencil.depthCompare, 'less-equal');
  assert.equal(record.pipelines.normal.fragment.targets[0].writeMask, 15);
  const a = f.stage(record, [10, 20]), b = f.stage(record, [30, 40]); f.owner.write();
  assert.ok(packet(a, 0).every(v => v === 10)); assert.ok(packet(a, 1).every(v => v === 20));
  assert.ok(packet(b, 0).every(v => v === 30)); assert.ok(packet(b, 1).every(v => v === 40));
  assert.equal(a.nodeProgram.groups[0].offset % 256, 0); assert.equal(b.nodeProgram.groups[1].offset % 256, 0);
  assert.equal(a.program, undefined); assert.equal(a.instanceCount, 1);
});
test('draws preserve native source index order and intersect current draw ranges', async t => {
  const f = rig(t), record = await f.add(); f.source.drawRange = {start: 1, count: 2};
  const command = f.stage(record, [1, 2], {first: 0, count: 2, frontFaceCW: true});
  assert.equal(command.first, 1); assert.equal(command.count, 1);
  assert.equal(command.pipeline, record.pipelines.flipped);
  assert.deepEqual(Array.from(new Uint16Array(command.indexBuffer.bytes.buffer).slice(0, 3)), [0, 1, 2]);
});
test('captured source ranges survive later drawRange changes while direct draws use the current range', async t => {
  const f = rig(t), record = await f.add(); f.source.drawRange = {start: 1, count: 1};
  const captured = f.stage(record, [1, 2], {first: 0, count: 3, nodeRangeCaptured: true});
  const direct = f.stage(record, [1, 2], {first: 0, count: 3});
  assert.deepEqual([captured.first, captured.count], [0, 3]);
  assert.deepEqual([direct.first, direct.count], [1, 1]);
});
test('invalid native draw ranges and flags fail without consuming uniform slots', async t => {
  const f = rig(t, {maxDraws: 1}), record = await f.add();
  for (const input of [{first: -1}, {first: NaN}, {count: .5}, {count: Infinity}, {frontFaceCW: 1},
    {nodeRangeCaptured: 'yes'}, {nodeRangeCaptured: true, first: 0}])
    assert.throws(() => f.stage(record, [1, 2], input), {code: 'ANIMATION_RENDER_VALUE'});
  assert.equal(f.owner.uniformDiagnostics.draws, 0); f.stage(record);
  assert.equal(f.owner.uniformDiagnostics.draws, 1);
});
test('later mutation or detachment cannot rewrite staged uniform packets', async t => {
  const f = rig(t), record = await f.add([16, 16]);
  const a = new Uint32Array([0x80000000, 0x7fc12345, 0x7f800000, 17]);
  const original = new Uint8Array(a.buffer).slice(), command = {};
  f.owner.stage(record, {nodeUniforms: [new Uint8Array(a.buffer), new Uint8Array(a.buffer)]}, command);
  a.fill(0); structuredClone(a.buffer, {transfer: [a.buffer]}); f.owner.write();
  assert.deepEqual(packet(command, 0), original); assert.deepEqual(packet(command, 1), original);
});
test('program registration snapshots shader, raster and group sizes before awaiting', async t => {
  const f = rig(t), release = f.holdPipelines(), program = f.program([16, 32]);
  const pending = f.add([], {nodeProgram: program});
  program.vertex = 'changed'; program.groups[0].byteLength = 4096; f.raster.writeMask = 0;
  release(); const record = await pending;
  assert.equal(record.groups[0].byteLength, 16); assert.equal(record.pipelines.normal.vertex.module.code, '// native vertex');
  assert.equal(record.pipelines.normal.fragment.targets[0].writeMask, 15);
});
test('uniform arena growth leaves earlier native groups and offsets valid', async t => {
  const f = rig(t, {maxBufferSize: 65536}), small = await f.add([16, 16]);
  const first = f.stage(small, [1, 2]), original = first.nodeProgram.groups.map(p => p.group);
  const large = await f.add([65536, 32768]), second = f.stage(large, [3, 4]); f.owner.write();
  assert.deepEqual(first.nodeProgram.groups.map(p => p.group), original);
  assert.ok(packet(first, 0).every(v => v === 1)); assert.ok(packet(first, 1).every(v => v === 2));
  assert.ok(packet(second, 0).every(v => v === 3)); assert.ok(packet(second, 1).every(v => v === 4));
  assert.ok(f.owner.uniformDiagnostics.allocatedBytes <= f.owner.uniformDiagnostics.maxBytes);
});
test('submission writes no native allocations and empty frames write no uniform bytes', async t => {
  const f = rig(t), record = await f.add(), allocations = [f.buffers.length, f.bindGroups.length];
  f.owner.begin(); f.owner.write(); const writes = f.writes.length;
  f.stage(record); f.owner.write();
  assert.deepEqual([f.buffers.length, f.bindGroups.length], allocations); assert.equal(f.writes.length, writes + 2);
  f.owner.begin(); f.owner.write(); assert.equal(f.writes.length, writes + 2);
});
test('malformed packets fail before consuming a draw slot', async t => {
  const f = rig(t, {maxDraws: 1}), record = await f.add([16, 32]);
  assert.throws(() => f.stage(record, [1], {nodeUniforms: [new Uint8Array(16), new Uint8Array(16)]}), {code: 'ANIMATION_RENDER_VALUE'});
  assert.throws(() => f.stage(record, [1], {nodeUniforms: [new Uint8Array(16), new Uint8Array(new SharedArrayBuffer(32))]}), {code: 'ANIMATION_RENDER_VALUE'});
  f.stage(record); assert.equal(f.owner.uniformDiagnostics.draws, 1);
  assert.throws(() => f.stage(record), {code: 'ANIMATION_RENDER_LIMIT'});
  f.owner.begin(); f.stage(record);
});
for (const empty of [false, true]) test(`disposal interrupts native pipeline preparation (${empty ? 'zero' : 'two'} groups)`, async t => {
  const f = rig(t), release = f.holdPipelines(), pending = f.add(empty ? [] : undefined);
  const rejected = assert.rejects(pending, {code: 'ANIMATION_RENDER_DISPOSED'});
  for (let i = 0; i < 16; i++) await Promise.resolve();
  f.owner.dispose(); await rejected; release(); assert.equal(f.owner.uniformDiagnostics.allocatedBytes, 0);
});
test('device loss interrupts native pipeline preparation and retains the loss reason', async t => {
  const f = rig(t), release = f.holdPipelines(), pending = f.add();
  const rejected = assert.rejects(pending, {code: 'ANIMATION_RENDER_LOST'});
  for (let i = 0; i < 16; i++) await Promise.resolve();
  f.lose({message: 'native fixture loss'}); await rejected; release();
  assert.throws(() => f.owner.begin(), /native fixture loss/);
});
for (const kind of ['pipeline', 'scope']) test(`${kind} errors cannot publish a native draw record`, async t => {
  const f = rig(t), error = Error(kind); f.errors({[kind]: error}); await assert.rejects(f.add(), e => e === error);
  f.errors(); const record = await f.add(); f.stage(record);
});
test('foreign or removed records and changed geometry are refused', async t => {
  const f = rig(t), record = await f.add();
  assert.throws(() => f.stage({...record}), {code: 'ANIMATION_RENDER_VALUE'});
  f.owner.remove(record); assert.throws(() => f.stage(record), {code: 'ANIMATION_RENDER_VALUE'});
  const next = await f.add(); f.gpu.dispose();
  assert.throws(() => f.stage(next), {code: 'GEOMETRY_GPU_DISPOSED'});
});
test('group budgets and device binding limits are checked before native allocation', async t => {
  const f = rig(t, {maxBindGroups: 2}), allocated = f.buffers.length;
  await assert.rejects(f.add([16, 16, 16]), {code: 'ANIMATION_RENDER_OPTIONS'});
  await assert.rejects(f.add([65540]), {code: 'ANIMATION_RENDER_LIMIT'});
  assert.equal(f.buffers.length, allocated);
  const descriptor = f.program([16]); descriptor.groups[0].index = 2;
  await assert.rejects(f.add([], {nodeProgram: descriptor}), {code: 'ANIMATION_RENDER_OPTIONS'});
});
