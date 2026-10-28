import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuRenderTarget} from '../gpu_render_target.mjs';
import {inspectGpuDepthCopy, submitGpuDepthCopy} from '../gpu_depth_transfer.mjs';
import {depthDevice, deferred, flush} from './depth_transfer_test_fixture.mjs';
const make = (f, options = {}) => createGpuRenderTarget(f.device, {
  width: 4, height: 4, format: 'rgba8unorm', depthFormat: 'depth32float',
  sampleDepth: true, copyDestination: true, cube: true, ...options,
});
function storage(target) {
  let depth, color;
  target.withSampledTextures((c, d) => { color = c; depth = d; });
  return {depth, color};
}
function setup(sourceOptions = {}, destinationOptions = {}) {
  const f = depthDevice(), source = make(f, sourceOptions), destination = make(f, destinationOptions);
  const a = storage(source), b = storage(destination);
  a.depth.data.set(Array.from(a.depth.data, (_, i) => (i + 1) / (a.depth.data.length + 1)));
  b.depth.data.fill(0.9); b.depth.stencil.fill(93); b.color.data.fill(0.4);
  return {...f, source, destination, a, b};
}
const code = value => ({code: `GPU_DEPTH_TRANSFER_${value}`});

for (const depthFormat of ['depth16unorm', 'depth24plus', 'depth32float', 'depth24plus-stencil8', 'depth32float-stencil8'])
  test(`${depthFormat}: six depth-face copies bind single-layer views and preserve color/stencil`, async () => {
    const f = setup({depthFormat}, {depthFormat}), beforeColor = f.b.color.data.slice();
    for (let face = 0; face < 6; face++) {
      assert.equal(f.destination.copyDepthFrom(f.source, {sourceFace: face, face: 5 - face}), true);
      const pass = f.submissions.at(-1)[0][0];
      assert.equal(pass.group.entries[0].resource.texture, f.a.depth);
      assert.deepEqual(pass.group.entries[0].resource.options, {
        dimension: '2d', baseArrayLayer: face, arrayLayerCount: 1, aspect: 'depth-only', baseMipLevel: 0, mipLevelCount: 1,
      });
      assert.equal(pass.desc.depthStencilAttachment.view.texture, f.b.depth);
      assert.equal(pass.desc.depthStencilAttachment.view.options.baseArrayLayer, 5 - face);
      assert.equal(pass.desc.depthStencilAttachment.view.options.arrayLayerCount, 1);
      assert.deepEqual(pass.desc.colorAttachments, []);
      assert.equal(pass.desc.depthStencilAttachment.depthLoadOp, 'load');
      if (depthFormat.endsWith('-stencil8')) assert.equal(pass.desc.depthStencilAttachment.stencilReadOnly, true);
    }
    assert.equal(f.buffers.length, 6); assert.ok(f.buffers.every(buffer => buffer.size === 32));
    assert.equal(f.pipelines.length, 1, 'face selection does not multiply pipelines');
    for (let face = 0; face < 6; face++) {
      const expected = f.a.depth.data.slice(face * 16, (face + 1) * 16);
      assert.deepEqual(f.b.depth.data.slice((5 - face) * 16, (6 - face) * 16), expected);
      assert.deepEqual(await f.destination.readDepthPixels({face: 5 - face}), expected);
    }
    assert.deepEqual(f.b.color.data, beforeColor); assert.ok(f.b.depth.stencil.every(value => value === 93));
    assert.equal(f.destination.diagnostics.depthCopies, 6); assert.equal(f.destination.diagnostics.mipGenerations, 0);
    await f.destination.whenIdle(); assert.ok(f.buffers.every(buffer => buffer.destroyCount === 1));
    f.source.dispose(); f.destination.dispose();
  });

for (const [sourceCube, destinationCube] of [[true, false], [false, true]])
  test(`depth copies ${sourceCube ? 'cube' : '2D'} -> ${destinationCube ? 'cube' : '2D'} with default 2D face`, async () => {
    const f = setup({cube: sourceCube}, {cube: destinationCube});
    for (let face = 0; face < 6; face++) {
      f.destination.copyDepthFrom(f.source, {...(sourceCube ? {sourceFace: face} : {}), ...(destinationCube ? {face} : {})});
      const expected = f.a.depth.data.slice(sourceCube ? face * 16 : 0, sourceCube ? (face + 1) * 16 : 16);
      assert.deepEqual(await f.destination.readDepthPixels(destinationCube ? {face} : {}), expected);
    }
    await f.destination.whenIdle(); f.source.dispose(); f.destination.dispose();
  });

test('flipped subrectangle addresses each face extent and leaves all other depths unchanged', async () => {
  const f = setup({width: 6, height: 6}), before = f.b.depth.data.slice();
  f.destination.copyDepthFrom(f.source, {sourceFace: 4, face: 2, sourceX: 2, sourceY: 1, x: 1, y: 0, width: 2, height: 2, flipY: true});
  const expected = before.slice();
  for (let row = 0; row < 2; row++) for (let column = 0; column < 2; column++)
    expected[2 * 16 + (row + 2) * 4 + column + 1] = f.a.depth.data[4 * 36 + (row + 3) * 6 + column + 2];
  assert.deepEqual(f.b.depth.data, expected);
  const read = await f.destination.readDepthPixels({face: 2, x: 1, y: 0, width: 2, height: 2, flipY: true});
  assert.deepEqual(read, Float32Array.of(expected[45], expected[46], expected[41], expected[42]));
  assert.deepEqual(f.submissions[0][0][0].scissor, [1, 2, 2, 2]);
  await f.destination.whenIdle(); f.source.dispose(); f.destination.dispose();
});

test('a 2D MSAA source requires an explicit sample before depth enters a cube face', async () => {
  const f = setup({cube: false, sampleCount: 4});
  assert.throws(() => f.destination.copyDepthFrom(f.source, {face: 3}), code('SAMPLES'));
  assert.equal(f.buffers.length, 0);
  f.destination.copyDepthFrom(f.source, {face: 3, sampleIndex: 2});
  assert.deepEqual(await f.destination.readDepthPixels({face: 3}), Float32Array.from({length: 16}, (_, i) => f.a.depth.data[i * 4 + 2]));
  assert.ok(f.b.depth.data.slice(0, 48).every(value => value === Math.fround(0.9)));
  assert.match(f.pipelines[0].desc.fragment.module.code, /texture_depth_multisampled_2d/);
  assert.match(f.pipelines[0].desc.fragment.module.code, /i32\(parameters.control.x\)/);
  await f.destination.whenIdle(); f.source.dispose(); f.destination.dispose();
});

test('faces, aliasing, budgets, precision changes and usage fail before GPU work and remain retryable', async () => {
  const f = setup(), data = {width: 4, height: 4, cube: true, sampleCount: 1, sampleDepth: true, depthFormat: 'depth32float'};
  for (const face of [undefined, null, -1, 6, 0.5, NaN, Infinity, '1', true]) {
    assert.throws(() => f.destination.copyDepthFrom(f.source, {sourceFace: face, face: 0}), code('FACE'));
    assert.throws(() => f.destination.copyDepthFrom(f.source, {sourceFace: 0, face}), code('FACE'));
  }
  for (const edit of [{maxBytes: 63}, {sourceX: 5}, {width: 5}, {sampleIndex: 1}, {mipLevel: 1}, {maxInFlightBytes: 0}])
    assert.throws(() => f.destination.copyDepthFrom(f.source, {sourceFace: 0, face: 0, ...edit}));
  const accessor = Object.defineProperty({sourceFace: 0}, 'face', {enumerable: true, get() { assert.fail('must not read getter'); }});
  assert.throws(() => f.destination.copyDepthFrom(f.source, accessor), code('OPTIONS'));
  assert.throws(() => f.source.copyDepthFrom(f.source, {sourceFace: 0, face: 1}), {code: 'GPU_TARGET_FEEDBACK'});
  assert.throws(() => inspectGpuDepthCopy({...data, cube: false}, data, {sourceFace: 1, face: 0}), code('FACE'));
  assert.throws(() => inspectGpuDepthCopy(data, {...data, cube: false}, {sourceFace: 0, face: 1}), code('FACE'));
  assert.throws(() => inspectGpuDepthCopy({...data, sampleCount: 4}, data, {sourceFace: 0, face: 1}), code('SOURCE'));
  assert.throws(() => inspectGpuDepthCopy({...data, width: 3}, data, {sourceFace: 0, face: 1}), code('SOURCE'));
  assert.throws(() => inspectGpuDepthCopy({...data, sampleDepth: false}, data, {sourceFace: 0, face: 1}), code('USAGE'));
  assert.throws(() => inspectGpuDepthCopy(data, {...data, depthFormat: 'depth16unorm'}, {sourceFace: 0, face: 1}), code('FORMAT'));
  const plan = inspectGpuDepthCopy(data, data, {sourceFace: 5, face: 4});
  assert.ok(Object.isFrozen(plan)); assert.equal(plan.sourceLayers, 6);
  assert.throws(() => submitGpuDepthCopy(f.device, {...f.a.depth, depthOrArrayLayers: 1}, f.b.depth.createView(), plan), code('SOURCE'));
  assert.equal(f.submissions.length, 0); assert.equal(f.buffers.length, 0); assert.equal(f.destination.failed, false);
  f.destination.copyDepthFrom(f.source, {sourceFace: 0, face: 1}); await f.destination.whenIdle();
  f.source.dispose(); f.destination.dispose();
});

test('cube depth format conversion remains explicit and stencil stays read-only', async () => {
  const f = setup({}, {depthFormat: 'depth24plus-stencil8'});
  assert.throws(() => f.destination.copyDepthFrom(f.source, {sourceFace: 2, face: 4}), code('FORMAT'));
  f.destination.copyDepthFrom(f.source, {sourceFace: 2, face: 4, allowFormatConversion: true});
  assert.equal(f.pipelines[0].desc.depthStencil.format, 'depth24plus-stencil8');
  assert.equal(f.pipelines[0].desc.depthStencil.stencilWriteMask, 0);
  assert.ok(f.b.depth.stencil.every(value => value === 93));
  await f.destination.whenIdle(); f.source.dispose(); f.destination.dispose();
});

test('depth jobs across cube faces share the tightest outstanding uniform budget', async () => {
  const f = setup(), wait = deferred(); f.control.fence = wait.promise;
  f.destination.copyDepthFrom(f.source, {sourceFace: 0, face: 1, maxInFlightBytes: 32});
  assert.throws(() => f.destination.copyDepthFrom(f.source, {sourceFace: 2, face: 3, maxInFlightBytes: 65536}), code('BUDGET'));
  assert.equal(f.destination.failed, false); assert.equal(f.buffers.length, 1);
  await flush(); assert.equal(f.buffers[0].destroyed, false);
  wait.resolve(); await f.destination.whenIdle();
  f.destination.copyDepthFrom(f.source, {sourceFace: 2, face: 3, maxInFlightBytes: 32}); await f.destination.whenIdle();
  assert.ok(f.buffers.every(buffer => buffer.destroyCount === 1)); assert.equal(f.pipelines.length, 1);
  f.source.dispose(); f.destination.dispose();
});

test('depth copies retain both cube generations and their uniform across resize', async () => {
  const f = setup(), wait = deferred(); f.control.fence = wait.promise;
  f.destination.copyDepthFrom(f.source, {sourceFace: 3, face: 5});
  f.source.resize(8, 8); f.destination.resize(8, 8);
  assert.equal(f.a.depth.destroyed, false); assert.equal(f.b.depth.destroyed, false);
  await flush(); assert.equal(f.buffers[0].destroyed, false);
  assert.deepEqual(f.b.depth.data.slice(80, 96), f.a.depth.data.slice(48, 64));
  wait.resolve(); await Promise.all([f.source.whenIdle(), f.destination.whenIdle()]);
  assert.equal(f.a.depth.destroyed, true); assert.equal(f.b.depth.destroyed, true);
  assert.equal(f.buffers[0].destroyCount, 1); f.source.dispose(); f.destination.dispose();
});

test('cube copy validation failure keeps the uniform until its work fence and evicts its pipeline', async () => {
  const f = setup(), wait = deferred(); f.control.fence = wait.promise;
  f.control.validation = {message: 'cube depth rejected'};
  f.destination.copyDepthFrom(f.source, {sourceFace: 1, face: 3});
  await flush(); assert.equal(f.buffers[0].destroyed, false);
  wait.resolve(); await assert.rejects(f.destination.whenIdle(), /cube depth rejected/);
  assert.equal(f.buffers[0].destroyCount, 1);
  const next = make(f); next.copyDepthFrom(f.source, {sourceFace: 1, face: 3}); await next.whenIdle();
  assert.equal(f.pipelines.length, 2); next.dispose(); f.source.dispose(); f.destination.dispose();
});

test('device loss interrupts pending cube depth work and releases the uniform once', async () => {
  const f = setup(), wait = deferred(); f.control.fence = wait.promise;
  f.destination.copyDepthFrom(f.source, {sourceFace: 4, face: 2});
  const idle = f.destination.whenIdle(); f.loss.resolve({message: 'cube device lost'});
  await assert.rejects(idle, /cube device lost/); await flush();
  assert.equal(f.buffers[0].destroyCount, 1); assert.equal(f.source.failed, true); assert.equal(f.destination.failed, true);
  wait.resolve(); await flush(); assert.equal(f.buffers[0].destroyCount, 1);
  f.source.dispose(); f.destination.dispose();
});

test('empty suspended cube depth copies still require faces and submit nothing', () => {
  const f = depthDevice(), source = make(f, {width: 0, height: 0}), destination = make(f, {width: 0, height: 0});
  assert.throws(() => destination.copyDepthFrom(source, {face: 0}), code('FACE'));
  assert.equal(destination.copyDepthFrom(source, {sourceFace: 5, face: 4}), false);
  assert.equal(f.buffers.length, 0); assert.equal(f.textures.length, 0); assert.equal(f.submissions.length, 0);
  source.dispose(); destination.dispose();
});


test('array depth transfers select independent layers and preserve other depth, color and stencil contents', async t => {
  const f = setup({cube: false, arrayLayers: 3, depthArray: true, depthFormat: 'depth24plus-stencil8'},
    {cube: false, arrayLayers: 5, depthArray: true, depthFormat: 'depth24plus-stencil8'});
  t.after(() => { f.source.dispose(); f.destination.dispose(); });
  const expected = f.b.depth.data.slice(), colors = f.b.color.data.slice();
  for (let sourceFace = 0; sourceFace < 3; sourceFace++) {
    const face = 4 - sourceFace;
    f.destination.copyDepthFrom(f.source, {sourceFace, face});
    expected.set(f.a.depth.data.subarray(sourceFace * 16, (sourceFace + 1) * 16), face * 16);
    const pass = f.submissions.at(-1)[0][0];
    assert.equal(pass.group.entries[0].resource.options.baseArrayLayer, sourceFace);
    assert.equal(pass.desc.depthStencilAttachment.view.options.baseArrayLayer, face);
    assert.equal(pass.desc.depthStencilAttachment.stencilReadOnly, true);
  }
  assert.deepEqual(f.b.depth.data, expected); assert.deepEqual(f.b.color.data, colors);
  assert.ok(f.b.depth.stencil.every(value => value === 93));
  assert.equal(f.pipelines.length, 1);
  assert.throws(() => f.destination.copyDepthFrom(f.source, {sourceFace: 3, face: 0}), code('FACE'));
  assert.throws(() => f.destination.copyDepthFrom(f.source, {sourceFace: 0, face: 5}), code('FACE'));
  assert.deepEqual(await f.destination.readDepthPixels({face: 4}), f.a.depth.data.slice(0, 16));
  await f.destination.whenIdle(); assert.ok(f.buffers.every(buffer => buffer.destroyCount === 1));
});

test('array color with shared 2D depth copies only its one depth attachment into an array layer', async t => {
  const f = setup({cube: false, arrayLayers: 3}, {cube: false, arrayLayers: 3, depthArray: true});
  t.after(() => { f.source.dispose(); f.destination.dispose(); });
  f.destination.copyDepthFrom(f.source, {face: 2});
  assert.deepEqual(await f.destination.readDepthPixels({face: 2}), f.a.depth.data);
  assert.throws(() => f.destination.copyDepthFrom(f.source, {sourceFace: 2, face: 0}), code('FACE'));
  await f.destination.whenIdle();
});

test('a selected private MSAA array depth layer receives every source sample without allocating an illegal MSAA array', async t => {
  const f = depthDevice(), source = make(f, {cube: false, sampleCount: 4}),
    destination = make(f, {cube: false, arrayLayers: 3, depthArray: true, sampleCount: 4, sampleDepth: false});
  t.after(() => { source.dispose(); destination.dispose(); });
  const input = storage(source).depth; input.data.set(Float32Array.from(input.data, (_, i) => i / 64));
  const outputs = [];
  for (let face = 0; face < 3; face++) destination.withFrame(frame => outputs.push(frame.depthView.texture), {face});
  assert.equal(new Set(outputs).size, 3);
  assert.ok(outputs.every(texture => texture.depthOrArrayLayers === 1 && texture.sampleCount === 4));
  destination.copyDepthFrom(source, {face: 2});
  assert.deepEqual(outputs[2].data, input.data);
  assert.ok(outputs[0].data.every(value => value === 0)); assert.ok(outputs[1].data.every(value => value === 0));
  assert.match(f.pipelines[0].desc.fragment.module.code, /@builtin\(sample_index\)/);
  await destination.whenIdle();
});
