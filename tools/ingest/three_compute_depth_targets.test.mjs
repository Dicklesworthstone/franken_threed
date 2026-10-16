import test from 'node:test';
import assert from 'node:assert/strict';
import {rig, depthTarget, DepthTexture, THREE, sampled, sampler, kernel, dispatches, plain} from './three_compute_targets_test_fixture.mjs';

test('native depth24plus, depth32float and packed depth/stencil retain scalar depth bindings', async () => {
  for (const [type, format, native] of [[THREE.UnsignedIntType, THREE.DepthFormat, 'depth24plus'],
    [THREE.FloatType, THREE.DepthFormat, 'depth32float'], [THREE.UnsignedInt248Type, THREE.DepthStencilFormat, 'depth24plus-stencil8']]) {
    const r = await rig(), target = depthTarget(type, format); r.renderer.produce(target, 7, 0.12345678);
    r.renderer.compute(kernel([sampled(target.depthTexture), sampler(target.depthTexture)]));
    const [, shader, groups, , values] = dispatches(r)[0];
    assert.deepEqual(values, [0.12345678]);
    assert.deepEqual(plain(shader.bindGroupLayouts[0].entries[0].texture), {sampleType: 'depth', viewDimension: '2d', multisampled: false});
    assert.equal(groups[0].entries[0].resource.nativeTexture, r.storages[0].depth);
    assert.equal(groups[0].entries[0].resource.nativeTexture.options.format, native);
    assert.equal(groups[0].entries[0].resource.descriptor.aspect, 'depth-only');
    assert.equal(groups[0].entries[0].resource.descriptor.mipLevelCount, 1);
    assert.equal(r.targets.capture(target).depthSample, null); assert.equal(r.targets.capture(target).sample, null);
    assert.equal(target.depthTexture.isRenderTargetTexture, false);
    await r.renderer.waitForGPU(); r.renderer.dispose();
  }
});
test('one compute dispatch reads both attachments of the same source target under one native guard', async () => {
  const r = await rig(), target = depthTarget(); r.renderer.produce(target, 11, 0.75);
  r.renderer.compute(kernel([sampled(target.depthTexture), sampled(target.texture), sampler(target.depthTexture), sampled(target.depthTexture)]));
  assert.deepEqual(dispatches(r)[0][4], [0.75, 11, 0.75]);
  const entries = dispatches(r)[0][2][0].entries;
  assert.equal(entries[0].resource, entries[3].resource); assert.notEqual(entries[0].resource, entries[1].resource);
  assert.equal(r.storages.length, 1); assert.equal(r.targets.allocatedBytes, 128);
  assert.ok(r.storages.every(s => !s.texture.borrowed && !s.depth.borrowed)); r.renderer.dispose();
});
test('depth-only compute does not regenerate color mips, while a later joint read does', async () => {
  const r = await rig(), target = depthTarget(); target.texture.generateMipmaps = true;
  r.renderer.produce(target, 19, 0.5); r.renderer.compute(kernel([sampled(target.depthTexture)]));
  assert.equal(r.events.filter(e => e[0] === 'mips').length, 0);
  r.renderer.compute(kernel([sampled(target.texture), sampled(target.depthTexture)]));
  assert.equal(r.events.filter(e => e[0] === 'mips').length, 1);
  const bindings = dispatches(r)[1][2][0].entries;
  assert.equal(bindings[0].resource.descriptor.mipLevelCount, 3); assert.equal(bindings[1].resource.descriptor.mipLevelCount, 1);
  r.renderer.compute(kernel([sampled(target.depthTexture)])); assert.equal(r.events.filter(e => e[0] === 'mips').length, 1);
  r.renderer.dispose();
});
test('queued depth reads follow producer order and retain their call-time samplers', async () => {
  const r = await rig(), target = depthTarget(), node = kernel([sampled(target.depthTexture), sampler(target.depthTexture)]);
  r.renderer._preparing = true;
  const a = r.renderer.produce(target, 3, 0.125), b = r.renderer.computeAsync(node);
  target.depthTexture.magFilter = target.depthTexture.minFilter = THREE.LinearFilter;
  const c = r.renderer.produce(target, 8, 0.875), d = r.renderer.computeAsync(node);
  await r.renderer.flush(); await Promise.all([a, b, c, d]);
  assert.deepEqual(dispatches(r).map(e => e[4]), [[0.125], [0.875]]);
  assert.deepEqual(dispatches(r).map(e => e[2][0].entries[1].resource.sampler.magFilter), ['nearest', 'linear']);
  assert.equal(r.device.features.size, 0); r.renderer.dispose();
});
test('color-depth layout replacement recompiles while retaining an earlier queued color view', async () => {
  const r = await rig(), target = depthTarget(), textureNode = {value: target.texture}; r.renderer.produce(target, 3, 0.6);
  const node = kernel([{...sampled(target.texture), textureNode}]); r.renderer._preparing = true;
  const a = r.renderer.computeAsync(node); textureNode.value = target.depthTexture; const b = r.renderer.computeAsync(node);
  await r.renderer.flush(); await Promise.all([a, b]);
  assert.deepEqual(dispatches(r).map(e => e[4]), [[3], [0.6]]);
  assert.deepEqual(dispatches(r).map(e => e[1].bindGroupLayouts[0].entries[0].texture.sampleType), ['float', 'depth']);
  assert.equal(r.builds, 2); r.renderer.dispose();
});
test('multi-target mixed color/depth inputs count each target once in aggregate admission', async () => {
  const r = await rig({maxBytes: 256}), a = depthTarget(), b = depthTarget();
  r.renderer.produce(a, 12, 0.25); r.renderer.produce(b, 24, 0.5);
  r.renderer.compute(kernel([sampled(a.depthTexture), sampled(b.texture), sampled(a.texture), sampled(b.depthTexture)]));
  assert.deepEqual(dispatches(r)[0][4], [0.25, 24, 12, 0.5]); assert.equal(r.targets.allocatedBytes, 256); r.renderer.dispose();
  const q = await rig({maxBytes: 255}), c = depthTarget(), d = depthTarget();
  assert.throws(() => q.renderer.compute(kernel([sampled(c.depthTexture), sampled(d.depthTexture)])), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(q.storages.length, 0); q.renderer.dispose();
});
test('depth disposal or replacement invalidates queued joint use before any native work', async () => {
  for (const change of [t => t.depthTexture.dispose(), t => { t.depthTexture = new DepthTexture(); t.depthTexture.renderTarget = t; }]) {
    const r = await rig(), target = depthTarget(); r.renderer._preparing = true;
    const task = r.renderer.computeAsync(kernel([sampled(target.texture), sampled(target.depthTexture)]));
    const rejected = assert.rejects(task, {code: 'THREE_TARGET_STALE'}); change(target);
    await r.renderer.flush(); await rejected; assert.equal(r.storages.length, 0); assert.equal(dispatches(r).length, 0); r.renderer.dispose();
  }
});
test('joint native borrow unwinds before disposal requested by a storage upload callback', async () => {
  const r = await rig(), target = depthTarget();
  const attribute = {isStorageBufferAttribute: true, array: new Float32Array(1), itemSize: 1, count: 1,
    onUploadCallback() { target.depthTexture.dispose(); }};
  assert.throws(() => r.renderer.compute(kernel([sampled(target.texture), sampled(target.depthTexture), {isStorageBuffer: true, attribute}])), {code: 'THREE_TARGET_STALE'});
  assert.equal(r.targets.allocatedBytes, 0); assert.equal(r.storages[0].disposed, true); assert.equal(dispatches(r).length, 0); r.renderer.dispose();
});
test('depth compilation prepares native views without a dispatch or a color mip pass', async () => {
  const r = await rig(), target = depthTarget(); target.texture.generateMipmaps = true;
  await r.renderer.compileComputeAsync(kernel([sampled(target.depthTexture)]));
  assert.equal(dispatches(r).length, 0); assert.equal(r.events.filter(e => e[0] === 'mips').length, 0);
  assert.equal(r.storages[0].depth.options.format, 'depth32float'); r.renderer.dispose();
});
test('comparison, MSAA depth and writable depth routes stay explicit unsupported inputs', async () => {
  for (const change of [t => { t.depthTexture.compareFunction = 515; }, t => { t.samples = 4; }]) {
    const r = await rig(), target = depthTarget(); change(target);
    assert.throws(() => r.renderer.compute(kernel([sampled(target.depthTexture)])), {code: 'THREE_TARGET_UNSUPPORTED'});
    assert.equal(r.storages.length, 0); r.renderer.dispose();
  }
  const r = await rig(), target = depthTarget();
  assert.throws(() => r.renderer.compute(kernel([{...sampled(target.depthTexture), store: true}])), {code: 'THREE_COMPUTE_TARGET_ACCESS'});
  assert.equal(r.storages.length, 0); r.renderer.dispose();
});
