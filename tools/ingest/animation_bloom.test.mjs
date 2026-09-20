import assert from 'node:assert/strict';
import test from 'node:test';
import { createGpuAnimationBloom, animationBloomPlan, animationBloomSettings, animationBloomCompositeWgsl } from './animation_bloom.mjs';
import { gpuPostprocessFixture, deferred } from './fixtures/animation/gpu_postprocess_fixture.mjs';
const code = value => ({ code: `ANIMATION_BLOOM_${value}` });
async function setup(options = {}) {
  const f = gpuPostprocessFixture();
  f.bloom = await createGpuAnimationBloom(f.device, options);
  f.frame = { source: f.texture(), target: f.texture() };
  return f;
}

test('pyramid includes odd borders, stops at 1x1, accounts for both half-float textures', () => {
  const p = animationBloomPlan(5, 3);
  assert.deepEqual(p.sizes, [{ width: 3, height: 2 }, { width: 2, height: 1 }, { width: 1, height: 1 }]);
  assert.equal(p.textureBytes, 144); assert.equal(p.allocatedBytes, 160);
  assert.ok(Object.isFrozen(p) && Object.isFrozen(p.sizes[0]));
  assert.equal(animationBloomPlan(1, 1).sizes.length, 1);
});

test('limits and settings reject malformed, overflowing and unsupported values', () => {
  for (const value of [0, -1, 1.5, NaN, Infinity, 32769, '4']) assert.throws(() => animationBloomPlan(value, 2), code('LIMIT'));
  for (const levels of [0, 7, 1.5]) assert.throws(() => animationBloomPlan(8, 8, { levels }), code('LIMIT'));
  assert.throws(() => animationBloomPlan(8, 8, { maxBytes: 16 }), code('LIMIT'));
  for (const input of [{ strength: NaN }, { strength: 65 }, { threshold: -1 }, { softKnee: 1.1 }, { threshold: 65505 }]) assert.throws(() => animationBloomSettings(input), code('VALUE'));
  assert.equal(animationBloomSettings({ strength: 0, softKnee: 0, threshold: 0 }).threshold, 0);
});

test('compiles explicit-layout HDR pipelines without filtering features or samplers', async () => {
  const f = await setup({ levels: 6 });
  assert.equal(f.calls.pipelines.length, 5);
  assert.ok(f.calls.pipelines.every(p => p.fragment.targets[0].format === 'rgba16float'));
  assert.equal(f.calls.buffers[0].descriptor.size, 16);
  assert.equal(f.calls.scopes.length, 0);
  const composite = f.calls.pipelines.at(-1).layout.bindGroupLayouts[0];
  assert.equal(composite.entries.filter(e => e.texture).length, 7);
  assert.ok(composite.entries.filter(e => e.texture).every(e => e.texture.sampleType === 'unfilterable-float'));
  assert.match(animationBloomCompositeWgsl(2), /bloom1/);
  assert.doesNotMatch(animationBloomCompositeWgsl(2), /bloom2/);
  f.bloom.dispose();
});

test('executes extract/downsample and ordered horizontal/vertical/composite passes in one submission', async () => {
  const f = await setup({ levels: 3 });
  assert.equal(f.bloom.render(f.frame), f.bloom);
  const commands = f.calls.submissions[0]; assert.equal(commands.length, 1);
  const passes = commands[0].passes;
  assert.deepEqual(passes.map(p => p.pipeline.fragment.entryPoint), [
    'extract_main', 'horizontal_main', 'vertical_main', 'downsample_main', 'horizontal_main', 'vertical_main',
    'downsample_main', 'horizontal_main', 'vertical_main', 'composite_main',
  ]);
  assert.ok(passes.every(p => p.vertices === 3));
  assert.equal(f.bloom.lastRender.passes, 10); assert.equal(f.bloom.allocatedBytes, 688);
  assert.deepEqual(f.calls.textures.map(t => [t.width, t.height]), [[8,4],[8,4],[4,2],[4,2],[2,1],[2,1]]);
  assert.equal(await f.bloom.whenIdle(), f.bloom); f.bloom.dispose();
});

test('unchanged extents and source reuse textures/pipelines/bind groups', async () => {
  const f = await setup({ levels: 2 }); f.bloom.render(f.frame);
  const groups = f.calls.groups.length; f.bloom.render({ ...f.frame, strength: 2 });
  assert.equal(f.calls.textures.length, 4); assert.equal(f.calls.groups.length, groups); assert.equal(f.calls.pipelines.length, 5);
  assert.equal(f.calls.submissions.length, 2); assert.equal(f.bloom.version, 2);
  const bytes = f.calls.writes.at(-1).bytes;
  assert.equal(new DataView(bytes.buffer).getFloat32(8, true), 2);
  f.bloom.render({ source: f.texture(), target: f.frame.target });
  assert.equal(f.calls.groups.length, groups + 2); assert.equal(f.calls.textures.length, 4);
  await f.bloom.whenIdle(); f.bloom.dispose();
});

test('zero strength is one copy/composite pass with no pyramid allocation even at a 16-byte budget', async () => {
  const f = await setup({ strength: 0, maxBytes: 16 }); f.bloom.render(f.frame);
  assert.equal(f.calls.textures.length, 0); assert.equal(f.bloom.allocatedBytes, 16);
  assert.equal(f.bloom.lastRender.passes, 1); assert.equal(f.bloom.lastRender.levels, 0);
  assert.equal(f.calls.submissions[0][0].passes[0].pipeline.fragment.entryPoint, 'composite_main');
  f.bloom.dispose();
});

test('single-pixel inputs bind unused levels safely and publish actual level count', async () => {
  const f = await setup({ levels: 6 }); f.bloom.render({ source: f.texture(1, 1), target: f.texture(1, 1) });
  assert.equal(f.bloom.lastRender.levels, 1); assert.equal(f.bloom.lastRender.passes, 4);
  const bytes = f.calls.writes[0].bytes;
  assert.equal(new DataView(bytes.buffer).getUint32(12, true), 1);
  f.bloom.dispose();
});

test('recoverable input failures issue no native writes and preserve the last successful snapshot', async () => {
  const f = await setup(); f.bloom.render(f.frame); const before = f.bloom.lastRender, writes = f.calls.writes.length;
  const badFrames = [
    { ...f.frame, strength: -1 }, { ...f.frame, unknown: true }, { ...f.frame, target: f.frame.source },
    { ...f.frame, target: f.texture(8, 8) }, { ...f.frame, source: f.texture(16, 8, 'rgba8unorm-srgb') },
    { ...f.frame, target: f.texture(16, 8, 'rgba8unorm') },
    { ...f.frame, source: { ...f.frame.source, sampleCount: 4 } },
    { ...f.frame, source: { ...f.frame.source, usage: 0 } },
  ];
  for (const frame of badFrames) assert.throws(() => f.bloom.render(frame));
  assert.equal(f.calls.writes.length, writes); assert.equal(f.bloom.lastRender, before); assert.equal(f.bloom.failed, false);
  f.bloom.render(f.frame); await f.bloom.whenIdle(); f.bloom.dispose();
});

test('resize checks old/new peak payload before allocation and leaves existing render usable', async () => {
  const f = await setup({ levels: 2, maxBytes: 900 });
  const small = { source: f.texture(8, 8), target: f.texture(8, 8) }; f.bloom.render(small);
  assert.equal(f.bloom.allocatedBytes, 336);
  assert.throws(() => f.bloom.render(f.frame), code('LIMIT'));
  assert.equal(f.calls.textures.length, 4); assert.ok(f.calls.textures.every(t => t.destroyed === 0));
  f.bloom.render(small); await f.bloom.whenIdle(); f.bloom.dispose();
});

test('successful resize releases old storage, replaces dimension-dependent bindings, and stays bounded', async () => {
  const f = await setup({ levels: 2, maxBytes: 4096 });
  f.bloom.render({ source: f.texture(8, 8), target: f.texture(8, 8) });
  const old = f.calls.textures.slice(); f.bloom.render(f.frame);
  assert.ok(old.every(t => t.destroyed === 1)); assert.equal(f.calls.textures.length, 8);
  assert.equal(f.bloom.allocatedBytes, 656); await f.bloom.whenIdle(); f.bloom.dispose();
});

test('partial allocation failure releases candidate and existing storage and makes native failure terminal', async () => {
  const f = await setup({ levels: 2 }); f.bloom.render(f.frame);
  f.controls.textureFailure = 6;
  assert.throws(() => f.bloom.render({ source: f.texture(32, 16), target: f.texture(32, 16) }), /allocation failure/);
  assert.equal(f.bloom.failed, true); assert.equal(f.bloom.allocatedBytes, 0);
  assert.ok(f.calls.textures.every(t => t.destroyed === 1));
  await assert.rejects(f.bloom.whenIdle(), /allocation failure/);
  f.bloom.dispose(); assert.equal(f.calls.deviceDestroyed, 0);
});

test('scoped asynchronous validation failure is observed by whenIdle and frees owned resources', async () => {
  const f = await setup(); f.controls.scopeError = { message: 'injected validation failure' };
  f.bloom.render(f.frame);
  await assert.rejects(f.bloom.whenIdle(), code('GPU'));
  assert.equal(f.bloom.failed, true); assert.equal(f.bloom.allocatedBytes, 0);
  assert.ok(f.calls.textures.every(t => t.destroyed === 1)); assert.equal(f.calls.scopes.length, 0);
  assert.equal(f.frame.source.destroyed, 0); assert.equal(f.frame.target.destroyed, 0); f.bloom.dispose();
});

test('shader compilation rejection aborts initialization without leaking uniform or textures', async () => {
  const f = gpuPostprocessFixture(); f.controls.pipelineFailure = new Error('shader rejected');
  await assert.rejects(createGpuAnimationBloom(f.device), /shader rejected/);
  assert.equal(f.calls.buffers.length, 0); assert.equal(f.calls.textures.length, 0); assert.equal(f.calls.scopes.length, 0);
});

test('abort before and during initialization does not publish an unusable bloom object', async () => {
  const a = gpuPostprocessFixture(), early = new AbortController(); early.abort(new Error('cancel early'));
  await assert.rejects(createGpuAnimationBloom(a.device, { signal: early.signal }), /cancel early/);
  assert.equal(a.calls.modules.length, 0);
  const b = gpuPostprocessFixture(), gate = deferred(), late = new AbortController(); b.controls.compile = gate;
  const pending = createGpuAnimationBloom(b.device, { signal: late.signal }); late.abort(new Error('cancel compile'));
  await assert.rejects(pending, /cancel compile/); gate.resolve(); await Promise.resolve();
  assert.equal(b.calls.buffers.length, 0); assert.equal(b.calls.textures.length, 0);
});

test('dispose cancels an outstanding completion wait and never destroys borrowed resources', async () => {
  const f = await setup(); f.bloom.render(f.frame); f.controls.completion = deferred();
  const idle = f.bloom.whenIdle(); f.bloom.dispose(); f.bloom.dispose();
  await assert.rejects(idle, code('DISPOSED')); assert.equal(f.bloom.allocatedBytes, 0);
  assert.ok(f.calls.textures.every(t => t.destroyed === 1)); assert.equal(f.calls.buffers[0].destroyed, 1);
  assert.equal(f.frame.source.destroyed + f.frame.target.destroyed + f.calls.deviceDestroyed, 0);
});

test('device loss rejects a pending completion wait and prevents further submissions', async () => {
  const f = await setup(); f.bloom.render(f.frame); f.controls.completion = deferred();
  const idle = f.bloom.whenIdle(); f.lost.resolve({ message: 'device unavailable' });
  await assert.rejects(idle, code('DEVICE_LOST')); assert.equal(f.bloom.failed, true);
  assert.throws(() => f.bloom.render(f.frame), code('DEVICE_LOST')); assert.equal(f.calls.submissions.length, 1);
  f.bloom.dispose();
});

test('source getters cannot dispose or recursively render during frame preparation', async () => {
  const f = await setup();
  for (const operation of [() => f.bloom.dispose(), () => f.bloom.render(f.frame)]) {
    assert.throws(() => f.bloom.render({ get source() { operation(); return f.frame.source; }, target: f.frame.target }), code('REENTRANT'));
  }
  assert.equal(f.calls.submissions.length, 0); assert.equal(f.bloom.disposed, false);
  f.bloom.render(f.frame); await f.bloom.whenIdle(); f.bloom.dispose();
});

test('device limits are checked before shader construction', async () => {
  const f = gpuPostprocessFixture(); f.device.limits.maxSampledTexturesPerShaderStage = 2;
  await assert.rejects(createGpuAnimationBloom(f.device, { levels: 3 }), code('DEVICE'));
  assert.equal(f.calls.modules.length, 0);
});
