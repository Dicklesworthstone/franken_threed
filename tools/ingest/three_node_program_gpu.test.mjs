// Recording GPU tests check ownership, native binding layout and byte history.
// Actual WGSL execution is compared with pinned r186 in the browser parity lane.
import test from 'node:test';
import assert from 'node:assert/strict';
import {createThreeNodeProgramMeshes} from './three_node_program_gpu.mjs';
import {createGpuProgramGeometry} from './gpu_buffer_geometry.mjs';
const fail = (code, message) => { throw Object.assign(Error(message), {code}); };
function rig(t, {maxDraws = 8, maxUniformBytes = 4 * 1024 * 1024, maxBindGroups = 4, maxBufferSize = 1 << 22, sampleCount = 1,
  depthFormat = 'depth24plus', configureGeometry = () => {}} = {}) {
  const buffers = [], bindGroups = [], pipelines = [], writes = [], shaderModules = [], pipelineLayouts = [];
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
    createBindGroupLayout: d => d,
    createPipelineLayout(d) { pipelineLayouts.push(d); return d; },
    createShaderModule(d) { shaderModules.push(d); return d; },
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
  configureGeometry(source, attributes);
  const gpu = createGpuProgramGeometry(device, source, attributes);
  const owner = createThreeNodeProgramMeshes({device, format: 'rgba8unorm', depthFormat,
    sampleCount, maxDraws, label: 'native-node-test', fail, scoped, maxUniformBytes, maxBindGroups});
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
  return {device, source, gpu, owner, raster, program, add, stage, buffers, bindGroups, pipelines, writes, shaderModules, pipelineLayouts,
    lose: resolveLost,
    holdPipelines() { pipelineGate = new Promise(resolve => { releasePipelines = resolve; }); return releasePipelines; },
    errors({pipeline = null, scope = null} = {}) { pipelineError = pipeline; scopeError = scope; }};
}
function packet(command, index) {
  const {group, offset} = command.nodeProgram.groups[index];
  const {buffer, size} = group.entries.find(entry => entry.resource.buffer).resource;
  assert.equal(group.layout, command.pipeline.layout.bindGroupLayouts[index]);
  return buffer.bytes.slice(offset, offset + size);
}
test('native alpha-to-coverage snapshots the source flag and only enables multisample pipelines', async t => {
  for (const sampleCount of [1, 4]) {
    const f = rig(t, {sampleCount}); f.raster.alphaToCoverageEnabled = true;
    const release = f.holdPipelines(), pending = f.add(); f.raster.alphaToCoverageEnabled = false; release();
    const record = await pending;
    assert.deepEqual(record.pipelines.normal.multisample, {count: sampleCount, alphaToCoverageEnabled: sampleCount > 1});
    assert.deepEqual(record.pipelines.flipped.multisample, record.pipelines.normal.multisample);
    f.raster.alphaToCoverageEnabled = 1;
    await assert.rejects(f.add(), {code: 'ANIMATION_RENDER_OPTIONS'});
  }
});
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
for (const [topology, native] of [['points', 'point-list'], ['lines', 'line-list'], ['line-strip', 'line-strip']])
  for (const Index of [null, Uint16Array, Uint32Array])
    test(`native ${topology} retains ${Index?.name ?? 'nonindexed'} geometry and primitive topology`, async t => {
      const f = rig(t, {configureGeometry(source) {
        if (Index === null) source.index = null;
        else source.index.array = new Index([2, 0, 1]);
      }});
      const stripIndexFormat = topology === 'line-strip' && Index ? Index === Uint16Array ? 'uint16' : 'uint32' : undefined;
      const record = await f.add([], {topology, stripIndexFormat}), command = f.stage(record, [], {first: 1, count: 2});
      assert.equal(record.pipelines.normal.primitive.topology, native);
      assert.equal(record.pipelines.flipped.primitive.topology, native);
      assert.equal(record.pipelines.normal.primitive.stripIndexFormat, stripIndexFormat);
      assert.deepEqual([command.first, command.count], [1, 2]);
      assert.equal(command.indexFormat, Index ? Index === Uint16Array ? 'uint16' : 'uint32' : null);
      if (Index) assert.deepEqual([...new Index(command.indexBuffer.bytes.buffer).subarray(0, 3)], [2, 0, 1]);
      else assert.equal(command.indexBuffer, null);
    });
test('native strip formats are validated before allocation and inferred for direct registrations', async t => {
  const f = rig(t), before = [f.buffers.length, f.pipelines.length];
  for (const options of [{topology: 'line-loop'}, {topology: 'toString'},
    {topology: 'points', stripIndexFormat: 'uint16'}, {topology: 'line-strip', stripIndexFormat: 'uint8'}])
    await assert.rejects(f.add([], options), {code: 'ANIMATION_RENDER_OPTIONS'});
  await assert.rejects(f.add([], {topology: 'line-strip', stripIndexFormat: 'uint32'}), {code: 'ANIMATION_RENDER_GEOMETRY'});
  assert.deepEqual([f.buffers.length, f.pipelines.length], before);
  const record = await f.add([], {topology: 'line-strip'});
  assert.equal(record.pipelines.normal.primitive.stripIndexFormat, 'uint16');
  f.source.index = {...f.source.index, array: new Uint32Array([0, 1, 2]), version: 0};
  f.gpu.update();
  assert.throws(() => f.stage(record, []), {code: 'ANIMATION_RENDER_GEOMETRY'});
  const replacement = await f.add([], {topology: 'line-strip', stripIndexFormat: 'uint32'});
  assert.equal(f.stage(replacement, []).indexFormat, 'uint32');
  assert.notEqual(replacement.pipelines.normal, record.pipelines.normal);
});
test('native primitive topology participates in pipeline sharing', async t => {
  const f = rig(t), records = [];
  for (const topology of ['triangles', 'lines', 'points', 'line-strip']) {
    const record = await f.add([], {topology}); records.push(record);
    assert.equal((await f.add([], {topology})).pipelines.normal, record.pipelines.normal);
  }
  assert.equal(new Set(records.map(record => record.pipelines.normal)).size, 4);
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
    {instanceCount: -1}, {instanceCount: .5}, {instanceCount: NaN}, {instanceCount: 0x100000000},
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

function texturedProgram(f, byteLength = 16) {
  const program = f.program([]);
  program.groups = [{index: 0, name: 'object', byteLength, entries: [
    {binding: 0, visibility: 2, sampler: {type: 'filtering'}},
    {binding: 1, visibility: 2, texture: {sampleType: 'float', viewDimension: '2d', multisampled: false}},
    {binding: 2, visibility: 7, buffer: {type: 'uniform', hasDynamicOffset: true, minBindingSize: byteLength}},
  ]}];
  return program;
}
const sampled = (view = {}, sampler = {}) => [{group: 0, binding: 0, resource: sampler}, {group: 0, binding: 1, resource: view}];

test('mixed native layouts preserve source visibility/indices and snapshot borrowed resources before preparation', async t => {
  const f = rig(t), program = texturedProgram(f), expected = structuredClone(program.groups[0].entries);
  const pending = f.add([], {nodeProgram: program});
  program.groups[0].entries[0].sampler.type = 'comparison'; program.groups[0].entries[2].visibility = 1;
  const record = await pending;
  assert.deepEqual(record.pipelines.normal.layout.bindGroupLayouts[0].entries, expected);
  const view = {}, sampler = {}, resources = sampled(view, sampler), allocations = [f.buffers.length, f.bindGroups.length],
    writes = f.writes.length;
  const command = f.stage(record, [17], {nodeResources: resources});
  resources[0].resource = {}; resources[1].resource = {}; resources.reverse();
  assert.deepEqual([f.buffers.length, f.bindGroups.length], allocations);
  assert.equal(command.nodeProgram.groups[0].group, null); assert.equal(f.writes.length, writes);
  f.owner.prepare(); f.owner.write();
  const group = command.nodeProgram.groups[0].group;
  assert.deepEqual(group.entries.map(entry => entry.binding), [0, 1, 2]);
  assert.equal(group.entries[0].resource, sampler); assert.equal(group.entries[1].resource, view);
  assert.equal(group.entries[2].resource.size, 16); assert.ok(packet(command, 0).every(v => v === 17));
  assert.equal(f.buffers.length, allocations[0]);
});

test('resource-only groups consume no uniform allocation or dynamic offset', async t => {
  const f = rig(t), program = f.program([]), before = f.buffers.length, view = {};
  program.groups = [
    {index: 0, byteLength: 0, entries: [{binding: 0, visibility: 1,
      texture: {sampleType: 'float', viewDimension: '2d', multisampled: false}}]},
    {index: 1, byteLength: 16, entries: [{binding: 0, visibility: 2,
      buffer: {type: 'uniform', hasDynamicOffset: true, minBindingSize: 16}}]},
  ];
  const record = await f.add([], {nodeProgram: program});
  assert.equal(f.buffers.length, before + 1); assert.equal(f.owner.uniformDiagnostics.groups, 1);
  const command = f.stage(record, [0, 19], {nodeResources: [{group: 0, binding: 0, resource: view}]});
  f.owner.prepare(); f.owner.write();
  assert.equal(command.nodeProgram.groups[0].offset, null); assert.equal(command.nodeProgram.groups[1].offset, 0);
  assert.deepEqual(command.nodeProgram.groups[0].group.entries, [{binding: 0, resource: view}]);
  assert.ok(packet(command, 1).every(v => v === 19));
});

test('per-draw texture changes reuse matching groups without recompilation or collapsing uniform histories', async t => {
  const f = rig(t), record = await f.add([], {nodeProgram: texturedProgram(f)});
  const a = sampled(), b = sampled(), pipelines = f.pipelines.length;
  const first = f.stage(record, [1], {nodeResources: a}), second = f.stage(record, [2], {nodeResources: b});
  const third = f.stage(record, [3], {nodeResources: a}); f.owner.prepare(); f.owner.write();
  assert.equal(first.nodeProgram.groups[0].group, third.nodeProgram.groups[0].group);
  assert.notEqual(first.nodeProgram.groups[0].group, second.nodeProgram.groups[0].group);
  assert.deepEqual([first, second, third].map(c => c.nodeProgram.groups[0].offset), [0, 256, 512]);
  assert.deepEqual([first, second, third].map(c => packet(c, 0)[0]), [1, 2, 3]);
  const groups = f.bindGroups.length; f.owner.begin();
  const next = f.stage(record, [4], {nodeResources: a}); f.owner.prepare(); f.owner.write();
  assert.equal(next.nodeProgram.groups[0].group, first.nodeProgram.groups[0].group);
  assert.equal(f.bindGroups.length, groups); assert.equal(f.pipelines.length, pipelines);
  f.owner.remove(record); assert.equal(record.disposed, true);
  assert.equal(a[0].resource.dead, undefined); assert.equal(a[1].resource.dead, undefined);
});

test('resource packet mismatches fail before uniform staging or any GPU allocation/write', async t => {
  const f = rig(t, {maxDraws: 1}), record = await f.add([], {nodeProgram: texturedProgram(f)});
  const allocations = [f.buffers.length, f.bindGroups.length, f.writes.length];
  for (const nodeResources of [undefined, [], sampled().slice(0, 1), [sampled()[0], sampled()[0]],
    [{group: 0, binding: 0, resource: {}}, {group: 1, binding: 1, resource: {}}],
    [{group: 0, binding: 0, resource: {}}, {group: 0, binding: 2, resource: {}}],
    [{group: 0, binding: 0, resource: {}}, {group: 0, binding: 1, resource: null}]])
    assert.throws(() => f.stage(record, [1], {nodeResources}), {code: 'ANIMATION_RENDER_VALUE'});
  assert.equal(f.owner.uniformDiagnostics.draws, 0);
  assert.deepEqual([f.buffers.length, f.bindGroups.length, f.writes.length], allocations);
  const command = f.stage(record, [2], {nodeResources: sampled()}); f.owner.prepare();
  assert.equal(command.nodeProgram.groups[0].offset, 0);
});

test('mixed bind group cache is bounded while stable resource selections retain their identity', async t => {
  const f = rig(t, {maxDraws: 1, maxBindGroups: 1}), record = await f.add([], {nodeProgram: texturedProgram(f)});
  const a = sampled(), b = sampled();
  const render = resources => { f.owner.begin(); const command = f.stage(record, [1], {nodeResources: resources});
    f.owner.prepare(); return command.nodeProgram.groups[0].group; };
  const first = render(a), second = render(b), groups = f.bindGroups.length;
  assert.equal(render(b), second); assert.equal(f.bindGroups.length, groups);
  assert.notEqual(render(a), first, 'the older resource group leaves the bounded cache');
  assert.equal(f.bindGroups.length, groups + 1);
});

test('unsupported native binding layouts fail before allocating uniform pages', async t => {
  const f = rig(t), before = f.buffers.length;
  for (const mutate of [
    p => { p.groups[0].entries[1].binding = 0; },
    p => { p.groups[0].entries[1].visibility = 8; },
    p => { p.groups[0].entries[2].buffer.minBindingSize = 32; },
    p => { p.groups[0].entries.push({binding: 3, visibility: 3, buffer: {type: 'uniform', hasDynamicOffset: true, minBindingSize: 16}}); },
  ]) {
    const program = texturedProgram(f); mutate(program);
    await assert.rejects(f.add([], {nodeProgram: program}), {code: 'ANIMATION_RENDER_OPTIONS'});
  }
  assert.equal(f.buffers.length, before);
});

test('read-only storage texture layouts retain exact format, dimension and borrowed views', async t => {
  const f = rig(t), before = f.buffers.length, writes = f.writes.length, program = f.program([]);
  program.groups = [{index: 0, byteLength: 0, entries: [
    {binding: 0, visibility: 1, storageTexture: {access: 'read-only', format: 'rgba8unorm', viewDimension: '2d'}},
    {binding: 1, visibility: 2, storageTexture: {access: 'read-only', format: 'rg32sint', viewDimension: '2d-array'}},
    {binding: 2, visibility: 3, storageTexture: {access: 'read-only', format: 'r32uint', viewDimension: '3d'}},
  ]}];
  const expected = structuredClone(program.groups[0].entries), pending = f.add([], {nodeProgram: program});
  program.groups[0].entries[0].storageTexture.format = 'rgba16float';
  const record = await pending;
  assert.deepEqual(record.pipelines.normal.layout.bindGroupLayouts[0].entries, expected);
  const resources = expected.map(entry => ({group: 0, binding: entry.binding, resource: {view: entry.binding}}));
  const command = f.stage(record, [], {nodeResources: resources});
  f.owner.prepare(); f.owner.write();
  assert.deepEqual(command.nodeProgram.groups[0].group.entries,
    resources.map(({binding, resource}) => ({binding, resource})));
  assert.equal(command.nodeProgram.groups[0].offset, null);
  assert.equal(f.buffers.length, before); assert.equal(f.writes.length, writes);
});

test('invalid graphics storage texture layouts and per-stage limits fail before pipeline allocation', async t => {
  const f = rig(t), before = f.buffers.length;
  const add = (entry = {}, storageTexture = {}) => {
    const program = f.program([]);
    program.groups = [{index: 0, byteLength: 0, entries: [{binding: 0, visibility: 3,
      storageTexture: {access: 'read-only', format: 'rgba8unorm', viewDimension: '2d', ...storageTexture}, ...entry}]}];
    return f.add([], {nodeProgram: program});
  };
  for (const storageTexture of [{access: 'write-only'}, {access: 'read-write'}, {format: 'bgra8unorm'},
    {format: 'rgba8unorm-srgb'}, {format: 'depth32float'}, {viewDimension: 'cube'}, {viewDimension: '1d'}])
    await assert.rejects(add({}, storageTexture), {code: 'ANIMATION_RENDER_OPTIONS'});
  for (const entry of [{visibility: 4}, {visibility: 7}, {sampler: {type: 'filtering'}}])
    await assert.rejects(add(entry), {code: 'ANIMATION_RENDER_OPTIONS'});
  f.device.limits.maxStorageTexturesPerShaderStage = 0;
  await assert.rejects(add(), {code: 'ANIMATION_RENDER_LIMIT'});
  assert.equal(f.buffers.length, before); assert.equal(f.pipelines.length, 0);
});

function storageProgram(f, {uniform = true, minBindingSize = 16, writable = false, visibility = writable ? 2 : 3} = {}) {
  const program = f.program([]);
  program.groups = [{index: 0, byteLength: uniform ? 16 : 0, entries: [
    {binding: 0, visibility, buffer: {type: writable ? 'storage' : 'read-only-storage', hasDynamicOffset: false, minBindingSize}},
    ...(uniform ? [{binding: 1, visibility: 3, buffer: {type: 'uniform', hasDynamicOffset: true, minBindingSize: 16}}] : []),
  ]}];
  return program;
}
const storageResource = (buffer, offset = 0, size = 16) => [{group: 0, binding: 0, resource: {buffer, offset, size}}];

test('native graphics borrows read-only storage alongside per-draw uniform slices without uploading it', async t => {
  const f = rig(t), program = storageProgram(f), record = await f.add([], {nodeProgram: program});
  const buffer = {size: 512, usage: 128}, resources = storageResource(buffer), before = [f.buffers.length, f.writes.length];
  const command = f.stage(record, [29], {nodeResources: resources});
  resources[0].resource.buffer = {}; resources[0].resource.offset = 256; resources[0].resource.size = 32;
  assert.deepEqual([f.buffers.length, f.writes.length], before);
  f.owner.prepare(); f.owner.write();
  const group = command.nodeProgram.groups[0].group;
  assert.deepEqual(group.entries[0], {binding: 0, resource: {buffer, offset: 0, size: 16}});
  assert.equal(group.entries[1].resource.size, 16);
  const uniform = group.entries[1].resource.buffer;
  assert.ok(uniform.bytes.slice(command.nodeProgram.groups[0].offset, command.nodeProgram.groups[0].offset + 16).every(v => v === 29));
  assert.equal(f.buffers.length, before[0]); assert.equal(f.writes.length, before[1] + 1);
  assert.ok(f.writes.every(write => write.buffer !== buffer));
});

test('storage-only native groups need no uniform arena or dynamic offset', async t => {
  const f = rig(t), before = f.buffers.length, record = await f.add([], {nodeProgram: storageProgram(f, {uniform: false})});
  const buffer = {size: 64, usage: 128}, command = f.stage(record, [], {nodeResources: storageResource(buffer)});
  f.owner.prepare(); f.owner.write();
  assert.equal(f.buffers.length, before); assert.equal(f.owner.uniformDiagnostics.groups, 0);
  assert.equal(command.nodeProgram.groups[0].offset, null);
  assert.deepEqual(command.nodeProgram.groups[0].group.entries, [{binding: 0, resource: {buffer, offset: 0, size: 16}}]);
});

test('storage bind groups cache allocation, offset and size instead of mutable wrapper identity', async t => {
  const f = rig(t), record = await f.add([], {nodeProgram: storageProgram(f, {uniform: false})});
  const buffer = {size: 512, usage: 128}, other = {size: 512, usage: 128};
  const commands = [[buffer, 0, 16], [buffer, 0, 16], [buffer, 256, 16], [buffer, 0, 32], [other, 0, 16]]
    .map(args => f.stage(record, [], {nodeResources: storageResource(...args)}));
  f.owner.prepare();
  const groups = commands.map(command => command.nodeProgram.groups[0].group);
  assert.equal(groups[0], groups[1]); assert.equal(new Set(groups).size, 4);
  const count = f.bindGroups.length; f.owner.begin();
  const next = f.stage(record, [], {nodeResources: storageResource(buffer)}); f.owner.prepare();
  assert.equal(next.nodeProgram.groups[0].group, groups[0]); assert.equal(f.bindGroups.length, count);
});

test('one storage allocation can serve aliased vertex and fragment bindings with sampled resources', async t => {
  const f = rig(t), program = texturedProgram(f), buffer = {size: 64, usage: 128};
  program.groups[0].entries.push(
    {binding: 3, visibility: 1, buffer: {type: 'read-only-storage'}},
    {binding: 4, visibility: 2, buffer: {type: 'read-only-storage'}});
  const record = await f.add([], {nodeProgram: program});
  const command = f.stage(record, [37], {nodeResources: [...sampled(),
    {group: 0, binding: 3, resource: {buffer}}, {group: 0, binding: 4, resource: {buffer}}]});
  f.owner.prepare();
  const entries = command.nodeProgram.groups[0].group.entries;
  assert.deepEqual(entries.map(entry => entry.binding), [0, 1, 2, 3, 4]);
  assert.deepEqual(entries[3].resource, {buffer, offset: 0, size: 64});
  assert.deepEqual(entries[4].resource, entries[3].resource);
});

test('storage ranges are validated before any uniform slot or GPU side effect is consumed', async t => {
  const f = rig(t, {maxDraws: 1}), record = await f.add([], {nodeProgram: storageProgram(f)}), buffer = {size: 512, usage: 128};
  const before = [f.buffers.length, f.bindGroups.length, f.writes.length];
  for (const resource of [{}, {buffer: {size: 512, usage: 32}}, {buffer, offset: -1, size: 16},
    {buffer, offset: 4, size: 16}, {buffer, offset: 256, size: 260}, {buffer, size: 8},
    {buffer, size: 18}, {buffer, size: 0}, {buffer, size: NaN}, {buffer, offset: Infinity, size: 16}])
    assert.throws(() => f.stage(record, [1], {nodeResources: [{group: 0, binding: 0, resource}]}), {code: 'ANIMATION_RENDER_VALUE'});
  assert.deepEqual([f.buffers.length, f.bindGroups.length, f.writes.length], before);
  assert.equal(f.owner.uniformDiagnostics.draws, 0);
  f.stage(record, [1], {nodeResources: storageResource(buffer)});
});

test('disjoint writable and read-only bindings still conflict across the whole buffer', async t => {
  const f = rig(t), program = storageProgram(f, {writable: true});
  program.groups[0].entries.push({binding: 2, visibility: 2,
    buffer: {type: 'read-only-storage', minBindingSize: 16}});
  const record = await f.add([], {nodeProgram: program}), buffer = {size: 512, usage: 128};
  const resources = [...storageResource(buffer), {group: 0, binding: 2, resource: {buffer, offset: 256, size: 16}}];
  const before = [f.buffers.length, f.bindGroups.length, f.writes.length];
  assert.throws(() => f.stage(record, [1], {nodeResources: resources}), {code: 'ANIMATION_RENDER_VALUE'});
  assert.deepEqual([f.buffers.length, f.bindGroups.length, f.writes.length], before);
  assert.equal(f.owner.uniformDiagnostics.draws, 0);
  resources[1].resource.buffer = {size: 512, usage: 128};
  f.stage(record, [2], {nodeResources: resources});
});

test('disjoint writable bindings and repeated draws are legal, while overlapping draw bindings are rejected', async t => {
  const f = rig(t), program = storageProgram(f, {writable: true});
  program.groups[0].entries.push({binding: 2, visibility: 2,
    buffer: {type: 'storage', minBindingSize: 16}});
  const record = await f.add([], {nodeProgram: program}), buffer = {size: 512, usage: 128};
  const resources = [...storageResource(buffer), {group: 0, binding: 2, resource: {buffer, offset: 0, size: 16}}];
  assert.throws(() => f.stage(record, [1], {nodeResources: resources}), {code: 'ANIMATION_RENDER_VALUE'});
  assert.equal(f.owner.uniformDiagnostics.draws, 0);
  resources[1].resource.offset = 256;
  const first = f.stage(record, [2], {nodeResources: resources}), second = f.stage(record, [3], {nodeResources: resources});
  f.owner.prepare(); f.owner.write();
  assert.equal(first.nodeProgram.groups[0].group, second.nodeProgram.groups[0].group);
  assert.deepEqual([first.nodeProgram.groups[0].offset, second.nodeProgram.groups[0].offset], [0, 256]);
  assert.deepEqual(first.nodeProgram.groups[0].group.entries.filter(entry => entry.binding !== 1).map(entry => entry.resource),
    [{buffer, offset: 0, size: 16}, {buffer, offset: 256, size: 16}]);
  assert.ok(f.writes.every(write => write.buffer !== buffer));
});

test('native draws share one buffer usage scope and begin resets it after rejection', async t => {
  const f = rig(t), writer = await f.add([], {nodeProgram: storageProgram(f, {writable: true})});
  const reader = await f.add([], {nodeProgram: storageProgram(f)}), buffer = {size: 512, usage: 128};
  for (const [first, second] of [[reader, writer], [writer, reader]]) {
    f.owner.begin();
    const before = [f.buffers.length, f.bindGroups.length, f.writes.length];
    f.stage(first, [1], {nodeResources: storageResource(buffer)});
    assert.throws(() => f.stage(second, [2], {nodeResources: storageResource(buffer, 256)}),
      {code: 'ANIMATION_RENDER_VALUE'});
    assert.equal(f.owner.uniformDiagnostics.draws, 1);
    assert.deepEqual([f.buffers.length, f.bindGroups.length, f.writes.length], before);
  }
  f.owner.begin();
  f.stage(reader, [3], {nodeResources: storageResource(buffer)});
  f.owner.prepare(); f.owner.write();
  assert.equal(f.owner.uniformDiagnostics.draws, 1);
});

test('storage layout and per-stage limits are separate from uniform limits', async t => {
  const f = rig(t), before = f.buffers.length;
  for (const buffer of [{type: 'storage'}, {type: 'read-only-storage', hasDynamicOffset: true},
    {type: 'read-only-storage', minBindingSize: 3}]) {
    const program = storageProgram(f); program.groups[0].entries[0].buffer = buffer;
    await assert.rejects(f.add([], {nodeProgram: program}), {code: 'ANIMATION_RENDER_OPTIONS'});
  }
  f.device.limits.maxStorageBufferBindingSize = 16;
  await assert.rejects(f.add([], {nodeProgram: storageProgram(f, {minBindingSize: 32})}), {code: 'ANIMATION_RENDER_LIMIT'});
  f.device.limits.maxUniformBuffersPerShaderStage = 0; f.device.limits.maxStorageBuffersPerShaderStage = 1;
  const program = storageProgram(f, {uniform: false});
  program.groups[0].entries.push({binding: 1, visibility: 1, buffer: {type: 'read-only-storage'}});
  await assert.rejects(f.add([], {nodeProgram: program}), {code: 'ANIMATION_RENDER_LIMIT'});
  assert.equal(f.buffers.length, before);
  program.groups[0].entries[0].visibility = 2;
  const record = await f.add([], {nodeProgram: program});
  assert.equal(record.groups[0].resources.length, 2); assert.equal(f.buffers.length, before);
  assert.throws(() => f.stage(record, [], {nodeResources: [
    {group: 0, binding: 0, resource: {buffer: {size: 32, usage: 128}}},
    {group: 0, binding: 1, resource: {buffer: {size: 16, usage: 128}}}]}), {code: 'ANIMATION_RENDER_LIMIT'});
});

function multiUniformProgram(f, {large = false, mixed = false} = {}) {
  const first = large ? 65536 : 16, second = large ? 16 : 32, offset = large ? 65536 : 256;
  const program = f.program([]);
  program.groups = [{index: 0, byteLength: offset + second,
    // Intentionally unordered: dynamic offsets follow binding numbers, while
    // packet ranges and resource-only slots retain their source indices.
    uniforms: [{binding: 5, offset, byteLength: second}, {binding: 1, offset: 0, byteLength: first}],
    entries: [
      {binding: 5, visibility: 2, buffer: {type: 'uniform', hasDynamicOffset: true, minBindingSize: second}},
      {binding: 1, visibility: 1, buffer: {type: 'uniform', hasDynamicOffset: true, minBindingSize: first}},
      ...(mixed ? [
        {binding: 0, visibility: 3, buffer: {type: 'read-only-storage', hasDynamicOffset: false, minBindingSize: 16}},
        {binding: 2, visibility: 2, texture: {sampleType: 'float', viewDimension: '2d', multisampled: false}},
        {binding: 3, visibility: 2, sampler: {type: 'filtering'}},
      ] : []),
    ]}];
  return program;
}
function uniformBytes(command, binding) {
  const packet = command.nodeProgram.groups[0], resource = packet.group.entries.find(entry => entry.binding === binding).resource;
  return resource.buffer.bytes.slice(packet.offset + resource.offset, packet.offset + resource.offset + resource.size);
}

test('multiple UBOs share a source group with sampled and storage resources without merging draw histories', async t => {
  const f = rig(t, {maxDraws: 2}), program = multiUniformProgram(f, {mixed: true});
  const pending = f.add([], {nodeProgram: program});
  program.groups[0].uniforms[0].offset = 0; program.groups[0].entries[0].buffer.minBindingSize = 64;
  const record = await pending, buffer = {size: 512, usage: 128}, view = {}, sampler = {}, writes = f.writes.length;
  const nodeResources = [{group: 0, binding: 0, resource: {buffer, offset: 256, size: 16}},
    {group: 0, binding: 2, resource: view}, {group: 0, binding: 3, resource: sampler}];
  const data = new Uint8Array(288); data.fill(11, 0, 16); data.fill(13, 256);
  const first = f.stage(record, [], {nodeUniforms: [data], nodeResources});
  data.fill(17, 0, 16); data.fill(19, 256);
  const second = f.stage(record, [], {nodeUniforms: [data], nodeResources});
  data.fill(0); structuredClone(data.buffer, {transfer: [data.buffer]}); nodeResources[0].resource.offset = 0;
  f.owner.prepare(); f.owner.write();
  const group = first.nodeProgram.groups[0].group;
  assert.equal(second.nodeProgram.groups[0].group, group);
  assert.deepEqual(group.entries.map(entry => entry.binding), [0, 1, 2, 3, 5]);
  assert.deepEqual(group.entries[0].resource, {buffer, offset: 256, size: 16});
  assert.equal(group.entries[2].resource, view); assert.equal(group.entries[3].resource, sampler);
  assert.deepEqual([group.entries[1], group.entries[4]].map(entry => [entry.resource.offset, entry.resource.size]), [[0, 16], [256, 32]]);
  assert.deepEqual([first, second].map(command => command.nodeProgram.groups[0].offsets), [[0, 0], [512, 512]]);
  for (const [command, a, b] of [[first, 11, 13], [second, 17, 19]]) {
    assert.ok(uniformBytes(command, 1).every(value => value === a));
    assert.ok(uniformBytes(command, 5).every(value => value === b));
  }
  assert.ok(f.writes.every(write => write.buffer !== buffer)); assert.equal(f.writes.length, writes + 1);
});

test('aggregate packets larger than 64 KiB retain valid individual UBO ranges and independent draws', async t => {
  const f = rig(t, {maxDraws: 2, maxBindGroups: 1, maxUniformBytes: 262144, maxBufferSize: 262144});
  const record = await f.add([], {nodeProgram: multiUniformProgram(f, {large: true})});
  const first = f.stage(record, [23]), second = f.stage(record, [31]); f.owner.prepare(); f.owner.write();
  assert.equal(record.groups[0].byteLength, 65552);
  assert.deepEqual([first, second].map(command => command.nodeProgram.groups[0].offsets), [[0, 0], [65792, 65792]]);
  for (const [command, expected] of [[first, 23], [second, 31]]) for (const binding of [1, 5])
    assert.ok(uniformBytes(command, binding).every(value => value === expected));
  const entries = first.nodeProgram.groups[0].group.entries;
  assert.deepEqual(entries.map(entry => entry.resource.size), [65536, 16]);
  assert.ok(f.bindGroups.every(group => group.entries.every(entry => !entry.resource.buffer || entry.resource.size <= 65536)));
  assert.ok(f.owner.uniformDiagnostics.allocatedBytes <= 262144);
});

test('malformed multi-UBO packet plans fail before allocating pages', async t => {
  const f = rig(t), before = f.buffers.length;
  for (const change of [
    group => { delete group.uniforms; },
    group => { group.uniforms.pop(); },
    group => { group.uniforms[0].binding = 1; },
    group => { group.uniforms[0].offset = 4; },
    group => { group.uniforms[0].offset = 0; },
    group => { group.uniforms[0].offset = 512; },
    group => { group.uniforms[0].byteLength = 16; },
    group => { group.byteLength += 4; },
  ]) {
    const program = multiUniformProgram(f); change(program.groups[0]);
    await assert.rejects(f.add([], {nodeProgram: program}), {code: 'ANIMATION_RENDER_OPTIONS'});
  }
  const program = multiUniformProgram(f, {large: true}); program.groups[0].entries[1].buffer.minBindingSize = 65540;
  await assert.rejects(f.add([], {nodeProgram: program}), {code: 'ANIMATION_RENDER_LIMIT'});
  assert.equal(f.buffers.length, before); assert.equal(f.bindGroups.length, 0);
});

test('the pipeline dynamic UBO limit counts bindings across source groups and shader stages', async t => {
  const f = rig(t), before = f.buffers.length;
  f.device.limits.maxDynamicUniformBuffersPerPipelineLayout = 1;
  f.device.limits.maxUniformBuffersPerShaderStage = 1;
  await assert.rejects(f.add([], {nodeProgram: multiUniformProgram(f)}), {code: 'ANIMATION_RENDER_LIMIT'});
  const program = f.program([16, 16]);
  program.groups.forEach((group, index) => { group.entries = [{binding: 0, visibility: index + 1,
    buffer: {type: 'uniform', hasDynamicOffset: true, minBindingSize: 16}}]; });
  await assert.rejects(f.add([], {nodeProgram: program}), {code: 'ANIMATION_RENDER_LIMIT'});
  assert.equal(f.buffers.length, before);
  f.device.limits.maxDynamicUniformBuffersPerPipelineLayout = 2;
  await f.add([], {nodeProgram: program});
});

function generatedInstanceRig(t, options = {}) {
  return rig(t, {...options, configureGeometry(source, attributes) {
    const matrix = {array: new Float32Array(48), stride: 16, version: 0, meshPerAttribute: 1,
      isInstancedInterleavedBuffer: true, updateRanges: [], clearUpdateRanges() { this.updateRanges.length = 0; }, onUploadCallback() {}};
    for (let instance = 0; instance < 3; instance++) {
      for (let diagonal = 0; diagonal < 4; diagonal++) matrix.array[instance * 16 + diagonal * 5] = 1;
      matrix.array[instance * 16 + 12] = instance + 1;
    }
    for (let column = 0; column < 4; column++) {
      const name = `nodeMatrix${column}`;
      source.attributes[name] = {data: matrix, offset: column * 4, itemSize: 4, count: 3,
        normalized: false, isInterleavedBufferAttribute: true};
      attributes.push({name, location: column + 1, locations: 1, components: 4, scalar: 'float'});
    }
    source.attributes.nodeColor = {array: new Float32Array([1, 0, 0, 0, 1, 0]), itemSize: 3, count: 2,
      version: 0, isInstancedBufferAttribute: true, meshPerAttribute: 1, updateRanges: [],
      clearUpdateRanges() { this.updateRanges.length = 0; }, onUploadCallback() {}};
    attributes.push({name: 'nodeColor', location: 5, locations: 1, components: 3, scalar: 'float'});
  }});
}

test('generated matrix columns and colors use instance streams with the smallest current capacity', async t => {
  const f = generatedInstanceRig(t, {maxDraws: 1}), program = {...f.program([16]), instanceCapacity: 3};
  const record = await f.add([], {nodeProgram: program}), layouts = record.pipelines.normal.vertex.buffers;
  assert.deepEqual(layouts.map(layout => [layout.stepMode, layout.arrayStride]), [['vertex', 12], ['instance', 64], ['instance', 12]]);
  assert.deepEqual(layouts[1].attributes.map(attribute => [attribute.shaderLocation, attribute.offset]), [[1, 0], [2, 16], [3, 32], [4, 48]]);
  assert.throws(() => f.stage(record, [1], {instanceCount: 3}), {code: 'ANIMATION_RENDER_VALUE'});
  assert.equal(f.owner.uniformDiagnostics.draws, 0);
  const first = f.stage(record, [1], {instanceCount: 2});
  assert.equal(first.instanceCount, 2); assert.equal(first.vertexBuffers.length, 3);
  assert.equal(new Float32Array(first.vertexBuffers[1].bytes.buffer)[28], 2);
  assert.deepEqual([...new Float32Array(first.vertexBuffers[2].bytes.buffer)], [1, 0, 0, 0, 1, 0]);
  f.owner.begin(); f.source.attributes.nodeColor.count = 1; f.gpu.update();
  assert.throws(() => f.stage(record, [2], {instanceCount: 2}), {code: 'ANIMATION_RENDER_VALUE'});
  assert.equal(f.stage(record, [2], {instanceCount: 1}).instanceCount, 1);
  f.owner.begin(); f.source.attributes.nodeColor.count = 0; f.gpu.update();
  assert.equal(f.stage(record, [3], {instanceCount: 0}).instanceCount, 0);
});

test('captured object capacities survive descriptor edits while uncapped Mesh.count remains available', async t => {
  const f = rig(t), allocated = f.buffers.length;
  for (const instanceCapacity of [-1, .5, NaN, Infinity, null, 0x100000000])
    await assert.rejects(f.add([16], {nodeProgram: {...f.program([16]), instanceCapacity}}), {code: 'ANIMATION_RENDER_OPTIONS'});
  assert.equal(f.buffers.length, allocated);
  const release = f.holdPipelines(), program = {...f.program([16]), instanceCapacity: 1};
  const pending = f.add([], {nodeProgram: program}); program.instanceCapacity = 100;
  release(); const record = await pending;
  assert.throws(() => f.stage(record, [1], {instanceCount: 2}), {code: 'ANIMATION_RENDER_VALUE'});
  assert.equal(f.owner.uniformDiagnostics.draws, 0); assert.equal(f.stage(record, [1]).instanceCount, 1);
  const ordinary = await f.add([16]);
  assert.equal(f.stage(ordinary, [2], {instanceCount: 19}).instanceCount, 19);
  const empty = await f.add([], {nodeProgram: {...f.program([16]), instanceCapacity: 0}});
  assert.throws(() => f.stage(empty, [3]), {code: 'ANIMATION_RENDER_VALUE'});
  assert.equal(f.stage(empty, [3], {instanceCount: 0}).instanceCount, 0);
});

test('native instance streams admit InstancedBufferGeometry while retaining their captured geometry kind', async t => {
  const f = generatedInstanceRig(t);
  f.source.isInstancedBufferGeometry = true; f.gpu.update();
  const instanced = await f.add([16]);
  assert.equal(f.stage(instanced, [1], {instanceCount: 2}).instanceCount, 2);
  assert.throws(() => f.stage(instanced, [1], {instanceCount: 3}), {code: 'ANIMATION_RENDER_VALUE'});
  f.source.isInstancedBufferGeometry = false; f.gpu.update();
  assert.throws(() => f.stage(instanced, [1], {instanceCount: 1}), {code: 'ANIMATION_RENDER_GEOMETRY'});
  f.owner.begin();
  const record = await f.add([16]);
  f.source.isInstancedBufferGeometry = true; f.gpu.update();
  assert.throws(() => f.stage(record, [1], {instanceCount: 1}), {code: 'ANIMATION_RENDER_GEOMETRY'});
  assert.equal(f.owner.uniformDiagnostics.draws, 0);
});

test('four thousand independent native records retain one compatible GPU program', async t => {
  const f = rig(t, {maxDraws: 4000}), records = [];
  for (let i = 0; i < 4000; i++) records.push(await f.add([16]));
  assert.equal(new Set(records).size, 4000);
  assert.equal(new Set(records.map(record => record.pipelines.normal)).size, 1);
  assert.equal(f.pipelines.length, 2); assert.equal(f.shaderModules.length, 2); assert.equal(f.pipelineLayouts.length, 1);
  const first = f.stage(records[0], [17]), last = f.stage(records[3999], [31]); f.owner.write();
  assert.ok(packet(first, 0).every(value => value === 17));
  assert.ok(packet(last, 0).every(value => value === 31));
  assert.notEqual(first.nodeProgram.groups[0].offset, last.nodeProgram.groups[0].offset);
});

test('shared native programs retain distinct geometry allocations and sampled draw histories', async t => {
  const f = rig(t), otherSource = {...f.source, attributes: {position: {...f.source.attributes.position,
    array: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0])}}, index: {...f.source.index, array: new Uint16Array([2, 1, 0])}};
  const otherGpu = createGpuProgramGeometry(f.device, otherSource,
    [{name: 'position', location: 0, locations: 1, components: 3, scalar: 'float'}]);
  t.after(() => otherGpu.dispose());
  const a = await f.add([], {nodeProgram: texturedProgram(f)});
  const b = await f.owner.add(otherGpu, {nodeProgram: texturedProgram(f), raster: f.raster});
  assert.equal(a.pipelines.normal, b.pipelines.normal); assert.equal(a.bindGroupLayouts, b.bindGroupLayouts);
  const resources = sampled(), first = f.stage(a, [11], {nodeResources: resources});
  const second = f.stage(b, [23], {nodeResources: resources}); f.owner.prepare(); f.owner.write();
  assert.notEqual(first.vertexBuffers[0], second.vertexBuffers[0]);
  assert.notEqual(first.indexBuffer, second.indexBuffer);
  assert.equal(first.nodeProgram.groups[0].group, second.nodeProgram.groups[0].group);
  assert.ok(packet(first, 0).every(value => value === 11));
  assert.ok(packet(second, 0).every(value => value === 23)); otherGpu.dispose();
});

test('native pipeline keys normalize binding order but distinguish shader and pipeline state', async t => {
  const f = rig(t, {sampleCount: 4}), base = await f.add([], {nodeProgram: texturedProgram(f)});
  const reordered = texturedProgram(f); reordered.groups[0].entries.reverse();
  assert.equal((await f.add([], {nodeProgram: reordered})).pipelines.normal, base.pipelines.normal);
  const changes = [
    (p, r) => { p.vertex += '\n// vertex variant'; },
    (p, r) => { p.fragment += '\n// fragment variant'; },
    (p, r) => { p.groups[0].entries[0].visibility = 1; },
    (p, r) => { p.groups[0].entries[1].texture.sampleType = 'unfilterable-float'; },
    (p, r) => { r.cullMode = 'none'; },
    (p, r) => { r.frontFace = 'cw'; },
    (p, r) => { r.writeMask = 7; },
    (p, r) => { r.depthWriteEnabled = false; },
    (p, r) => { r.depthCompare = 'greater'; },
    (p, r) => { r.depthBias = 1; },
    (p, r) => { r.blend = {color: {srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha'}}; },
    (p, r) => { r.alphaToCoverageEnabled = true; },
  ];
  for (const change of changes) {
    const program = texturedProgram(f), raster = {...f.raster}; change(program, raster);
    const record = await f.add([], {nodeProgram: program, raster});
    assert.notEqual(record.pipelines.normal, base.pipelines.normal);
  }
  assert.notEqual((await f.add([], {nodeProgram: texturedProgram(f), topology: 'lines'})).pipelines.normal, base.pipelines.normal);
  assert.equal(f.pipelines.length, 2 * (changes.length + 2));
});

test('native pipeline sharing compares vertex layouts rather than geometry identity', async t => {
  const f = rig(t), first = await f.add([16]), source = {...f.source, attributes: {position: {
    isInterleavedBufferAttribute: true, itemSize: 3, count: 3, offset: 0, normalized: false,
    data: {array: new Float32Array(12), stride: 4, version: 0, updateRanges: [], clearUpdateRanges() {}, onUploadCallback() {}},
  }}};
  const gpu = createGpuProgramGeometry(f.device, source,
    [{name: 'position', location: 0, locations: 1, components: 3, scalar: 'float'}]);
  t.after(() => gpu.dispose());
  const other = await f.owner.add(gpu, {nodeProgram: f.program([16]), raster: f.raster});
  assert.notEqual(first.pipelines.normal, other.pipelines.normal);
  assert.deepEqual([first, other].map(record => record.pipelines.normal.vertex.buffers[0].arrayStride), [12, 16]); gpu.dispose();
});

test('draw capacities, blend constants and stencil references do not split compatible native pipelines', async t => {
  const f = rig(t, {depthFormat: 'depth24plus-stencil8'});
  const raster = {...f.raster, blend: {color: {srcFactor: 'constant', dstFactor: 'one-minus-constant'}},
    blendConstant: [1, 0, 0, 1], stencil: {front: {passOp: 'replace'}}, stencilReference: 3};
  const a = await f.add([], {nodeProgram: {...f.program([16]), instanceCapacity: 1}, raster});
  const b = await f.add([], {nodeProgram: {...f.program([16]), instanceCapacity: 5},
    raster: {...raster, blendConstant: [0, 0, 1, 1], stencilReference: 7}});
  assert.equal(a.pipelines.normal, b.pipelines.normal); assert.equal(f.pipelines.length, 2);
  assert.throws(() => f.stage(a, [1], {instanceCount: 2}), {code: 'ANIMATION_RENDER_VALUE'});
  const first = f.stage(a, [1], {count: 1}), second = f.stage(b, [2], {count: 3, instanceCount: 5});
  assert.deepEqual([first.count, second.count, second.instanceCount], [1, 3, 5]);
  assert.deepEqual([first.blendConstant, second.blendConstant], [[1, 0, 0, 1], [0, 0, 1, 1]]);
  assert.deepEqual([first.stencilReference, second.stencilReference], [3, 7]);
  const other = await f.add([16], {raster: {...raster, stencil: {front: {passOp: 'invert'}}}});
  assert.notEqual(other.pipelines.normal, a.pipelines.normal);
});

test('shared layouts retain each UBO member offset in the resource group identity', async t => {
  const f = rig(t), programA = multiUniformProgram(f), programB = multiUniformProgram(f);
  programB.groups[0].uniforms[0].offset = 512; programB.groups[0].byteLength = 544;
  const a = await f.add([], {nodeProgram: programA}), b = await f.add([], {nodeProgram: programB});
  assert.equal(a.pipelines.normal, b.pipelines.normal); assert.equal(a.bindGroupLayouts, b.bindGroupLayouts);
  const bytesA = new Uint8Array(288), bytesB = new Uint8Array(544);
  bytesA.fill(11, 0, 16); bytesA.fill(13, 256);
  bytesB.fill(17, 0, 16); bytesB.fill(19, 512);
  const first = f.stage(a, [], {nodeUniforms: [bytesA]}), second = f.stage(b, [], {nodeUniforms: [bytesB]});
  f.owner.prepare(); f.owner.write();
  assert.notEqual(first.nodeProgram.groups[0].group, second.nodeProgram.groups[0].group);
  for (const [command, value1, value5] of [[first, 11, 13], [second, 17, 19]]) {
    assert.ok(uniformBytes(command, 1).every(value => value === value1));
    assert.ok(uniformBytes(command, 5).every(value => value === value5));
  }
});

test('concurrent registrations share pipeline preparation and evict a failed program before retry', async t => {
  const f = rig(t), release = f.holdPipelines(), failure = Error('shared pipeline failure');
  f.errors({pipeline: failure});
  const results = Promise.allSettled([f.add([]), f.add([]), f.add([])]);
  for (let i = 0; i < 16; i++) await Promise.resolve();
  assert.equal(f.pipelines.length, 2); assert.equal(f.shaderModules.length, 2); release();
  for (const result of await results) { assert.equal(result.status, 'rejected'); assert.equal(result.reason, failure); }
  f.errors(); const a = await f.add([]), b = await f.add([]);
  assert.equal(a.pipelines.normal, b.pipelines.normal); assert.equal(f.pipelines.length, 4);
});

test('removing one shared record preserves peers while the final removal releases its cache entry', async t => {
  const f = rig(t), program = texturedProgram(f), a = await f.add([], {nodeProgram: program});
  const b = await f.add([], {nodeProgram: program}), resources = sampled();
  const first = f.stage(a, [11], {nodeResources: resources}); f.owner.prepare();
  f.owner.remove(a); f.owner.remove(a);
  const second = f.stage(b, [17], {nodeResources: resources}); f.owner.prepare(); f.owner.write();
  assert.equal(second.nodeProgram.groups[0].group, first.nodeProgram.groups[0].group);
  assert.ok(packet(second, 0).every(value => value === 17));
  assert.equal((await f.add([], {nodeProgram: program})).pipelines.normal, b.pipelines.normal);
  f.owner.remove(b);
  const c = await f.add([32]); const d = await f.add([32]);
  f.owner.remove(c); f.owner.remove(d);
  const before = f.pipelines.length, replacement = await f.add([32]);
  assert.notEqual(replacement.pipelines.normal, c.pipelines.normal); assert.equal(f.pipelines.length, before + 2);
});
