/** Pipeline descriptors only; program_mrt_targets.mjs executes the translated shaders. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {createProgramMeshes} from './animation_program_mesh.mjs';
import {createGpuProgramGeometry} from './gpu_buffer_geometry.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

function fixture(t, formats = ['rgba8unorm', 'rgba16float'], samples = 1) {
  const device = geometryDevice(), pipelines = [];
  device.features = new Set();
  device.createRenderPipelineAsync = async descriptor => {pipelines.push(descriptor); return descriptor;};
  device.createRenderPipeline = descriptor => {pipelines.push(descriptor); return descriptor;};
  const gpu = createGpuProgramGeometry(device, {attributes: {}, drawRange: {start: 0, count: 3}}, []);
  const owner = createProgramMeshes({device, format: formats[0], colorFormats: formats, depthFormat: null,
    sampleCount: samples, maxDraws: 8, label: 'mrt-program', lost: device.lost,
    fail(code, message) {throw Object.assign(Error(message), {code});},
    scoped(_device, operation) {return {value: operation(), errors: Promise.resolve()};}});
  const program = {vertex: '// recorded', fragment: '// recorded',
    reflection: {textures: [], outputs: [{location: 0}, {location: 1}], uniformBufferSize: 16}};
  const raster = {frontFace: 'ccw', cullMode: 'none', blend: null, writeMask: 15};
  t.after(() => {owner.dispose(); gpu.dispose();});
  return {device, owner, gpu, program, raster, pipelines,
    add() {return owner.add(gpu, {program, raster});}};
}

for (const samples of [1, 4]) test(`ESSL programs bind both output locations at ${samples} samples`, async t => {
  const formats = ['rgba8unorm', 'rgba16float'], f = fixture(t, formats, samples);
  formats[1] = 'rgba32float'; // attachment descriptors were accepted at construction
  const pending = f.add();
  f.program.reflection.outputs[1].location = 7; f.raster.writeMask = 0;
  await pending;
  assert.equal(f.pipelines.length, 2);
  for (const pipeline of f.pipelines) {
    assert.deepEqual(pipeline.fragment.targets, [{format: 'rgba8unorm', writeMask: 15}, {format: 'rgba16float', writeMask: 15}]);
    assert.equal(pipeline.multisample.count, samples);
  }
});

test('sparse output locations disable writes and blending on unattached shader outputs', async t => {
  const f = fixture(t, ['rgba32float', 'rgba8unorm']);
  f.program.reflection.outputs = [{location: 1}];
  f.raster.blend = {color: {srcFactor: 'one', dstFactor: 'one', operation: 'add'},
    alpha: {srcFactor: 'one', dstFactor: 'zero', operation: 'add'}};
  await f.add();
  assert.deepEqual(f.pipelines[0].fragment.targets[0], {format: 'rgba32float', writeMask: 0});
  assert.deepEqual(f.pipelines[0].fragment.targets[1].blend, f.raster.blend);
});

test('every active MRT format validates blending before allocating a pipeline', async t => {
  const f = fixture(t, ['rgba8unorm', 'rgba32float']);
  f.raster.blend = {color: {srcFactor: 'one', dstFactor: 'one', operation: 'add'},
    alpha: {srcFactor: 'one', dstFactor: 'zero', operation: 'add'}};
  await assert.rejects(f.add(), /rgba32float blending requires float32-blendable/);
  assert.equal(f.pipelines.length, 0);
});

test('typed ESSL outputs preserve signed and unsigned attachment formats', async t => {
  const f = fixture(t, ['rgba32uint', 'rg32sint']);
  f.program.reflection.outputs = [{location: 0, glslType: 'uvec4'}, {location: 1, glslType: 'ivec2'}];
  await f.add();
  assert.deepEqual(f.pipelines[0].fragment.targets, [{format: 'rgba32uint', writeMask: 15}, {format: 'rg32sint', writeMask: 15}]);
});

for (const type of ['vec4', 'ivec4', undefined]) test(`unsigned ESSL attachments reject ${type ?? 'missing'} output typing before allocation`, async t => {
  const f = fixture(t, ['rgba32uint']);
  f.program.reflection.outputs = [{location: 0, glslType: type}];
  const buffers = f.device.buffers.length;
  await assert.rejects(f.add(), /require u32 fragment outputs/);
  assert.equal(f.pipelines.length, 0); assert.equal(f.device.buffers.length, buffers);
});

test('ESSL attachment outputs require every stored component', async t => {
  const f = fixture(t, ['rgba32uint']); f.program.reflection.outputs = [{location: 0, glslType: 'uvec2'}];
  await assert.rejects(f.add(), /require at least 4 fragment output components/);
  assert.equal(f.pipelines.length, 0);
});

test('invalid or duplicate fragment locations reject before program preparation', async t => {
  const f = fixture(t);
  for (const locations of [[0, 2], [0, 0], [-1], [0.5]]) {
    f.program.reflection.outputs = locations.map(location => ({location}));
    await assert.rejects(f.add(), /distinct allocated color attachments/);
  }
  assert.equal(f.pipelines.length, 0);
});

test('immediate callback-created MRT programs use the same attachment mapping', t => {
  const f = fixture(t);
  const record = f.owner.addImmediate(f.gpu, {program: f.program, raster: f.raster});
  assert.deepEqual(record.pipelines.normal.fragment.targets.map(target => target.format), ['rgba8unorm', 'rgba16float']);
});
