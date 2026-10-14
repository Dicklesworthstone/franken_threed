import test from 'node:test';
import assert from 'node:assert/strict';
import {createGpuRenderTarget} from '../gpu_render_target.mjs';
import {inspectGpuDepthCopy, submitGpuDepthCopy} from '../gpu_depth_transfer.mjs';
import {depthDevice, deferred, flush} from './depth_transfer_test_fixture.mjs';
const shape = (overrides = {}) => ({width: 4, height: 3, depthFormat: 'depth32float', sampleCount: 1, sampleDepth: true, ...overrides});
const code = suffix => error => error?.code?.endsWith(suffix);
function targets(f, sourceOptions = {}, destinationOptions = {}) {
  const common = {width: 4, height: 3, format: 'rgba8unorm', depthFormat: 'depth32float', sampleDepth: true, copyDestination: true};
  const source = createGpuRenderTarget(f.device, {...common, ...sourceOptions});
  const destination = createGpuRenderTarget(f.device, {...common, ...destinationOptions});
  let a, b;
  source.withDepthTexture(texture => { a = texture; });
  destination.withFrame(frame => { b = frame.depthView?.texture; });
  if (a) a.data.set(Array.from(a.data, (_, i) => (i + 1) / (a.data.length + 1)));
  if (b) { b.data.fill(0.9); b.stencil.fill(93); }
  return {source, destination, a, b};
}

test('depth planner captures both origins and an immutable bounded region', () => {
  const input = {sourceX: 1, sourceY: 1, x: 2, y: 1, width: 2, height: 2, flipY: true};
  const plan = inspectGpuDepthCopy(shape({height: 5}), shape({width: 6, height: 7}), input);
  input.sourceX = 99;
  assert.equal(plan.sourceX, 1); assert.equal(plan.sourceY, 2); assert.equal(plan.y, 4);
  assert.equal(plan.mode, 'single'); assert.ok(Object.isFrozen(plan));
  assert.throws(() => submitGpuDepthCopy({}, null, null, {...plan}), code('PLAN'));
});

test('planner rejects unsafe options, unsupported profiles and implicit precision/sample changes', () => {
  for (const input of [null, [], {width: -1}, {width: NaN}, {height: Infinity}, {x: 4.5}, {sourceX: 5},
    {width: 5}, {x: 3, width: 2}, {flipY: 1}, {mipLevel: 1}, {maxBytes: 1}, {maxInFlightBytes: 0},
    {allowFormatConversion: 'yes'}, {[Symbol('x')]: 1}, Object.create({x: 1})])
    assert.throws(() => inspectGpuDepthCopy(shape(), shape(), input));
  let reads = 0;
  assert.throws(() => inspectGpuDepthCopy(shape(), shape(), {get x() { reads++; return 0; }}));
  assert.equal(reads, 0);
  assert.throws(() => inspectGpuDepthCopy(shape({sampleDepth: false}), shape()), code('USAGE'));
  assert.throws(() => inspectGpuDepthCopy(shape(), shape({depthFormat: null})), code('FORMAT'));
  assert.throws(() => inspectGpuDepthCopy(shape(), shape({depthFormat: 'depth16unorm'})), code('FORMAT'));
  assert.equal(inspectGpuDepthCopy(shape(), shape({depthFormat: 'depth16unorm'}), {allowFormatConversion: true}).format, 'depth16unorm');
  assert.throws(() => inspectGpuDepthCopy(shape({sampleCount: 4}), shape()), code('SAMPLES'));
  assert.throws(() => inspectGpuDepthCopy(shape(), shape({sampleCount: 4})), code('SAMPLES'));
  for (const sampleIndex of [-1, 4, NaN, 0.5, null])
    assert.throws(() => inspectGpuDepthCopy(shape({sampleCount: 4}), shape(), {sampleIndex}), code('SAMPLES'));
  assert.throws(() => inspectGpuDepthCopy(shape({sampleCount: 4}), shape({sampleCount: 4}), {sampleIndex: 0}), code('SAMPLES'));
});

for (const depthFormat of ['depth16unorm', 'depth24plus', 'depth32float', 'depth24plus-stencil8', 'depth32float-stencil8']) {
  test(`native ${depthFormat} copy submits depth-only pass and preserves untouched regions`, async () => {
    const f = depthDevice(), {source, destination, a, b} = targets(f, {depthFormat}, {depthFormat});
    const color = f.textures.find(texture => texture.label === 'Offscreen render target color');
    color.data.fill(0.4); const savedColor = color.data.slice(), before = b.data.slice();
    assert.equal(destination.copyDepthFrom(source, {sourceX: 1, sourceY: 1, x: 0, y: 0, width: 2, height: 1}), true);
    assert.equal(f.submissions.length, 1, 'submission happens before any await');
    assert.equal(b.data[0], a.data[5]); assert.equal(b.data[1], a.data[6]);
    assert.deepEqual(b.data.slice(2), before.slice(2)); assert.deepEqual(color.data, savedColor);
    assert.ok(b.stencil.every(value => value === 93));
    const pass = f.submissions[0][0][0], pipeline = pass.pipeline.desc;
    assert.deepEqual(pass.desc.colorAttachments, []); assert.deepEqual(pipeline.fragment.targets, []);
    assert.equal(pass.desc.depthStencilAttachment.depthLoadOp, 'load');
    assert.equal(pass.desc.depthStencilAttachment.depthStoreOp, 'store');
    assert.equal(pipeline.depthStencil.depthCompare, 'always'); assert.equal(pipeline.depthStencil.depthWriteEnabled, true);
    assert.equal(pass.group.entries[0].resource.options.aspect, 'depth-only');
    if (depthFormat.endsWith('-stencil8')) {
      assert.equal(pass.desc.depthStencilAttachment.stencilReadOnly, true);
      assert.equal(pipeline.depthStencil.stencilWriteMask, 0);
      assert.equal(pass.desc.depthStencilAttachment.stencilLoadOp, undefined);
    }
    assert.match(pipeline.fragment.module.code, /@builtin\(frag_depth\)/);
    assert.match(pipeline.fragment.module.code, /vec2i\(p.xy\) - vec2i\(parameters.origins.zw\) \+ vec2i\(parameters.origins.xy\)/);
    assert.equal(destination.diagnostics.depthCopies, 1);
    await destination.whenIdle();
    assert.equal(f.buffers[0].destroyCount, 1); assert.equal(f.scopes.length, 0);
    source.dispose(); destination.dispose();
  });
}

test('native MSAA copies preserve all four samples or select an explicitly requested sample', async () => {
  const f = depthDevice(), {source, destination, a, b} = targets(f, {sampleCount: 4}, {sampleCount: 4});
  destination.copyDepthFrom(source);
  assert.deepEqual(b.data, a.data);
  assert.match(f.pipelines[0].desc.fragment.module.code, /@builtin\(sample_index\) sample: u32/);
  assert.match(f.pipelines[0].desc.fragment.module.code, /textureLoad\(source, pixel, i32\(sample\)\)/);
  assert.equal(f.pipelines[0].desc.multisample.count, 4);
  const single = createGpuRenderTarget(f.device, {width: 4, height: 3, format: 'rgba8unorm', depthFormat: 'depth32float', sampleDepth: true, copyDestination: true});
  assert.throws(() => single.copyDepthFrom(source), code('SAMPLES'));
  single.copyDepthFrom(source, {sampleIndex: 2});
  single.withDepthTexture(texture => assert.deepEqual(texture.data, Float32Array.from({length: 12}, (_, i) => a.data[i * 4 + 2])));
  assert.match(f.pipelines[1].desc.fragment.module.code, /i32\(parameters.control.x\)/);
  await Promise.all([destination.whenIdle(), single.whenIdle()]);
});

test('native identity, device, usage and live-generation gates run before allocation', () => {
  const f = depthDevice(), {source, destination} = targets(f);
  assert.throws(() => destination.copyDepthFrom(destination), code('FEEDBACK'));
  assert.throws(() => destination.copyDepthFrom({}), code('SOURCE'));
  const foreign = targets(depthDevice()).source;
  assert.throws(() => destination.copyDepthFrom(foreign), code('DEVICE'));
  const unsampled = createGpuRenderTarget(f.device, {depthFormat: 'depth32float'});
  assert.throws(() => destination.copyDepthFrom(unsampled), code('USAGE'));
  assert.throws(() => unsampled.copyDepthFrom(source), code('USAGE'));
  source.withDepthTexture(() => assert.throws(() => destination.copyDepthFrom(source), code('REENTRANT')));
  destination.withFrame(() => assert.throws(() => destination.copyDepthFrom(source), code('REENTRANT')));
  source.dispose(); assert.throws(() => destination.copyDepthFrom(source), code('DISPOSED'));
  assert.equal(f.buffers.length, 0); assert.equal(f.submissions.length, 0);
});

test('empty depth transfers have no GPU allocation or submission', () => {
  const f = depthDevice(), {source, destination} = targets(f);
  assert.equal(destination.copyDepthFrom(source, {width: 0}), false);
  assert.equal(destination.diagnostics.depthCopies, 0); assert.equal(f.buffers.length, 0);
  const empty = createGpuRenderTarget(f.device, {width: 0, height: 0, depthFormat: 'depth32float', sampleDepth: true});
  assert.equal(destination.copyDepthFrom(empty), false); assert.equal(f.submissions.length, 0);
});

test('uniform budgets are shared across targets and cannot relax an unfinished caller ceiling', async () => {
  const f = depthDevice(), {source, destination} = targets(f), wait = deferred();
  f.control.fence = wait.promise;
  destination.copyDepthFrom(source, {maxInFlightBytes: 32});
  assert.throws(() => source.copyDepthFrom(destination, {maxInFlightBytes: 65536}), code('BUDGET'));
  assert.equal(source.failed, false); assert.equal(f.buffers.length, 1);
  await flush(); assert.equal(f.buffers[0].destroyed, false);
  wait.resolve(); await destination.whenIdle();
  source.copyDepthFrom(destination, {maxInFlightBytes: 32}); await source.whenIdle();
  assert.equal(f.buffers.length, 2); assert.ok(f.buffers.every(buffer => buffer.destroyCount === 1));
  assert.equal(f.pipelines.length, 1, 'cached by device/profile, not owner or region');
});

test('depth-only writes do not regenerate otherwise clean color mipmaps', async () => {
  const f = depthDevice(), {source, destination} = targets(f, {}, {generateMipmaps: true});
  destination.withTexture(() => {}); const generations = destination.diagnostics.mipGenerations;
  destination.copyDepthFrom(source); destination.withTexture(() => {});
  assert.equal(destination.diagnostics.mipGenerations, generations);
  await destination.whenIdle();
});

test('resize after synchronous submission retains old borrowed attachments until its fence', async () => {
  const f = depthDevice(), {source, destination, a, b} = targets(f), wait = deferred();
  f.control.fence = wait.promise;
  destination.copyDepthFrom(source); source.resize(8, 6); destination.resize(8, 6);
  assert.equal(a.destroyed, false); assert.equal(b.destroyed, false);
  assert.equal(source.diagnostics.retiredTargets, 1); assert.equal(destination.diagnostics.retiredTargets, 1);
  wait.resolve(); await Promise.all([source.whenIdle(), destination.whenIdle()]);
  assert.equal(a.destroyed, true); assert.equal(b.destroyed, true);
  assert.equal(destination.width, 8); assert.equal(f.buffers[0].destroyCount, 1);
});

test('validation failures wait for pending work before releasing uniforms and evict cached pipelines', async () => {
  const f = depthDevice(), {source, destination} = targets(f), wait = deferred();
  f.control.fence = wait.promise; f.control.validation = {message: 'invalid depth program'};
  destination.copyDepthFrom(source);
  await flush(); assert.equal(f.buffers[0].destroyed, false);
  wait.resolve(); await assert.rejects(destination.whenIdle(), /invalid depth program/);
  assert.equal(destination.failed, true); assert.equal(f.buffers[0].destroyCount, 1);
  const next = targets(f); next.destination.copyDepthFrom(next.source); await next.destination.whenIdle();
  assert.equal(f.pipelines.length, 2);
});

for (const stage of ['shader', 'pipeline', 'buffer', 'write', 'group', 'encoder', 'begin', 'draw', 'end', 'finish', 'submit', 'pop', 'fence']) {
  test(`native depth copy cleans up and reports ${stage} failure`, async () => {
    const f = depthDevice(), {source, destination} = targets(f);
    f.control.fault = stage;
    destination.copyDepthFrom(source);
    await assert.rejects(destination.whenIdle(), new RegExp(`injected ${stage}`));
    assert.equal(destination.failed, true); assert.equal(f.scopes.length, 0);
    assert.ok(f.buffers.every(buffer => buffer.destroyCount === 1));
  });
}

test('device loss completes copies even when the native work fence never settles', async () => {
  const f = depthDevice(), {source, destination} = targets(f), wait = deferred();
  f.control.fence = wait.promise; destination.copyDepthFrom(source);
  const idle = destination.whenIdle(); f.loss.resolve({message: 'device gone'});
  await assert.rejects(idle, /device gone/); await flush();
  assert.equal(f.buffers[0].destroyCount, 1); assert.equal(source.failed, true); assert.equal(destination.failed, true);
});

test('source allocation validation is a dependency of the depth copy completion', async () => {
  const f = depthDevice(); f.control.validation = {message: 'source allocation invalid'};
  const {source, destination} = targets(f); destination.copyDepthFrom(source);
  await assert.rejects(destination.whenIdle(), /source allocation invalid/);
});

test('buffer limits and malformed borrowed depth metadata are rejected before GPU allocation', () => {
  const f = depthDevice(), {a, b} = targets(f), plan = inspectGpuDepthCopy(shape(), shape());
  f.device.limits.maxUniformBufferBindingSize = 16;
  assert.throws(() => submitGpuDepthCopy(f.device, a, b.createView(), plan), code('BUDGET'));
  f.device.limits.maxUniformBufferBindingSize = 65536;
  assert.throws(() => submitGpuDepthCopy(f.device, {...a, usage: 16}, b.createView(), plan), code('SOURCE'));
  assert.equal(f.buffers.length, 0);
});
