import test from 'node:test';
import assert from 'node:assert/strict';
import {rig, RenderTarget, THREE, sampled, sampler, kernel, dispatches, plain, deferred} from './three_compute_targets_test_fixture.mjs';

test('public compute samples the native color produced by rendering, without a replica or CPU upload', async () => {
  const r = await rig(), target = new RenderTarget(); r.renderer.produce(target, 17);
  r.renderer.compute(kernel([sampled(target.texture), sampler(target.texture), {...sampled({isStorageTexture: true}), store: true}]));
  const [, shader, groups, , values] = dispatches(r)[0];
  assert.deepEqual(values, [17]); assert.equal(groups[0].entries[0].resource.nativeTexture, r.storages[0].texture);
  assert.equal(groups[0].entries[2].resource.storageView, true);
  assert.deepEqual(plain(shader.bindGroupLayouts[0].entries[0].texture), {sampleType: 'float', viewDimension: '2d', multisampled: false});
  assert.equal(r.targets.capture(target).sample, null); assert.equal(r.targets.allocatedBytes, 64);
  assert.deepEqual(r.events.filter(e => e[0] === 'written'), [['written', 1]]);
  await r.renderer.waitForGPU(); r.renderer.dispose();
});
test('render-compute-render-compute FIFO observes both producer versions and call-time samplers', async () => {
  const r = await rig(), target = new RenderTarget(), node = kernel([sampled(target.texture), sampler(target.texture)]);
  r.renderer._preparing = true;
  const firstRender = r.renderer.produce(target, 3), first = r.renderer.computeAsync(node, [2, 1, 1]);
  target.texture.minFilter = target.texture.magFilter = THREE.LinearFilter;
  const secondRender = r.renderer.produce(target, 9), second = r.renderer.computeAsync(node, [5, 1, 1]);
  target.texture.wrapS = THREE.RepeatWrapping;
  assert.equal(r.storages.length, 0); assert.equal(dispatches(r).length, 0);
  await r.renderer.flush(); await Promise.all([firstRender, first, secondRender, second]);
  assert.deepEqual(dispatches(r).map(e => e[4]), [[3], [9]]);
  assert.deepEqual(dispatches(r).map(e => e[3][0]), [2, 5]);
  assert.deepEqual(dispatches(r).map(e => e[2][0].entries[1].resource.sampler.magFilter), ['nearest', 'linear']);
  assert.deepEqual(dispatches(r).map(e => e[2][0].entries[1].resource.sampler.addressModeU), ['clamp-to-edge', 'clamp-to-edge']);
  assert.equal(r.builds, 1); r.renderer.dispose();
});
test('four-sample source metadata selects the resolved primary texture, not a multisampled binding', async () => {
  const r = await rig(), target = new RenderTarget(); target.samples = 4;
  r.renderer.produce(target, 8); r.renderer.compute(kernel([sampled(target.texture)]));
  assert.deepEqual(r.events.find(e => e[0] === 'sampleData')[1], {samples: 4, primarySamples: 1, isMSAA: true});
  assert.equal(dispatches(r)[0][1].bindGroupLayouts[0].entries[0].texture.multisampled, false);
  assert.equal(r.storages[0].texture.options.sampleCount, 4); r.renderer.dispose();
});
test('R/RG/RGBA float32 inputs use explicit unfilterable layouts without precision loss', async () => {
  for (const format of [THREE.RedFormat, THREE.RGFormat, THREE.RGBAFormat]) {
    const r = await rig(), target = new RenderTarget(); target.texture.type = THREE.FloatType; target.texture.format = format;
    r.renderer.compute(kernel([sampled(target.texture), sampler(target.texture)]));
    const entries = dispatches(r)[0][1].bindGroupLayouts[0].entries;
    assert.equal(entries[0].texture.sampleType, 'unfilterable-float'); assert.equal(entries[1].sampler.type, 'non-filtering');
    assert.equal(dispatches(r)[0][2][0].entries[1].resource.sampler.mipmapFilter, 'nearest');
    assert.match(r.storages[0].texture.options.format, /32float$/); r.renderer.dispose();
  }
});
test('float32 filtering is feature-gated before any allocation', async () => {
  for (const enabled of [false, true]) {
    const r = await rig({features: enabled ? ['float32-filterable'] : []}), target = new RenderTarget();
    target.texture.type = THREE.FloatType; target.texture.minFilter = target.texture.magFilter = THREE.LinearFilter;
    const node = kernel([sampled(target.texture), sampler(target.texture)]);
    if (!enabled) { assert.throws(() => r.renderer.compute(node), {code: 'THREE_TARGET_SAMPLER'}); assert.equal(r.storages.length, 0); }
    else { r.renderer.compute(node); assert.equal(dispatches(r)[0][1].bindGroupLayouts[0].entries[0].texture.sampleType, 'float'); }
    r.renderer.dispose();
  }
});
test('mip generation precedes compute and repeated bindings share one native borrow/view', async () => {
  const r = await rig(), target = new RenderTarget(8, 4); target.texture.generateMipmaps = true;
  target.texture.minFilter = THREE.NearestMipmapNearestFilter;
  r.renderer.produce(target, 6); r.renderer.compute(kernel([sampled(target.texture), sampled(target.texture), sampler(target.texture)]));
  const groups = dispatches(r)[0][2];
  assert.equal(groups[0].entries[0].resource, groups[0].entries[1].resource);
  assert.equal(groups[0].entries[0].resource.descriptor.mipLevelCount, 4);
  assert.equal(groups[0].entries[2].resource.sampler.lodMaxClamp, 3);
  assert.ok(r.events.findIndex(e => e[0] === 'mips') < r.events.findIndex(e => e[0] === 'dispatch'));
  assert.equal(r.events.filter(e => e[0] === 'mips').length, 1); r.renderer.dispose();
});
test('all inputs stay borrowed through one multi-target dispatch', async () => {
  const r = await rig(), a = new RenderTarget(), b = new RenderTarget();
  r.renderer.produce(a, 12); r.renderer.produce(b, 24); r.renderer.compute(kernel([sampled(a.texture), sampled(b.texture)]));
  assert.deepEqual(dispatches(r)[0][4], [12, 24]); assert.ok(r.storages.every(s => !s.texture.borrowed)); r.renderer.dispose();
});
test('aggregate admission refuses multi-target allocation before creating any storage', async () => {
  const r = await rig({maxBytes: 100}), a = new RenderTarget(), b = new RenderTarget();
  assert.throws(() => r.renderer.compute(kernel([sampled(a.texture), sampled(b.texture)])), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(r.storages.length, 0); assert.equal(dispatches(r).length, 0); r.renderer.dispose();
});
test('disposal and resize invalidate captured target generations instead of silently recapturing', async () => {
  for (const change of [t => t.dispose(), t => t.texture.dispose(), t => t.setSize(8, 8), t => { t.width = 16; }]) {
    const r = await rig(), target = new RenderTarget(); r.renderer._preparing = true;
    const pending = r.renderer.computeAsync(kernel([sampled(target.texture)])); const rejected = assert.rejects(pending, {code: 'THREE_TARGET_STALE'});
    change(target); await r.renderer.flush(); await rejected;
    assert.equal(dispatches(r).length, 0); assert.equal(r.storages.length, 0); r.renderer.dispose();
  }
});
test('live texture-node replacement captures old and new targets without stale binding aliases', async () => {
  const r = await rig(), a = new RenderTarget(), b = new RenderTarget(); r.renderer.produce(a, 4); r.renderer.produce(b, 7);
  const textureNode = {value: a.texture}, node = kernel([{...sampled(a.texture), textureNode}]); r.renderer._preparing = true;
  const first = r.renderer.computeAsync(node); textureNode.value = b.texture; const second = r.renderer.computeAsync(node);
  await r.renderer.flush(); await Promise.all([first, second]); assert.deepEqual(dispatches(r).map(e => e[4]), [[4], [7]]); r.renderer.dispose();
});
test('native target compilation prepares bindings but never dispatches a kernel', async () => {
  const r = await rig(), target = new RenderTarget(), node = kernel([sampled(target.texture)]), progress = [];
  await r.renderer.compileComputeAsync(node, event => progress.push(event.loaded));
  assert.deepEqual(progress, [1]); assert.equal(dispatches(r).length, 0); assert.equal(r.storages.length, 1);
  r.renderer.compute(node); assert.equal(dispatches(r).length, 1); r.renderer.dispose();
});
test('target producer errors reach compute wait, readback and buffer-to-target copy completion', async () => {
  const r = await rig(), target = new RenderTarget(); r.renderer.produce(target, 1);
  const gate = deferred(); r.storages[0].wait(gate.promise);
  r.renderer.compute(kernel([sampled(target.texture)]));
  const error = new Error('target mip generation validation failed'); gate.reject(error);
  await assert.rejects(r.renderer.waitForGPU(), e => e === error);
  const attribute = {isStorageBufferAttribute: true, array: new Float32Array(1), itemSize: 1, count: 1};
  await assert.rejects(r.renderer.getArrayBufferAsync(attribute), e => e === error);
  await assert.rejects(r.renderer.copyStorageBufferToRenderTargetAsync(attribute, new RenderTarget()), e => e === error);
  r.renderer.dispose();
});
test('target binding scopes reject native validation and synchronous driver errors', async () => {
  const r = await rig(), target = new RenderTarget(); r.scopeErrors.push({message: 'bad native view'});
  r.renderer.compute(kernel([sampled(target.texture)])); await assert.rejects(r.renderer.waitForGPU(), {code: 'THREE_COMPUTE_TARGET_GPU'});
  r.renderer.dispose();
  const q = await rig(), other = new RenderTarget(), error = new Error('sampler allocation failed'); q.failSampler(error);
  assert.throws(() => q.renderer.compute(kernel([sampler(other.texture)])), e => e === error);
  assert.equal(dispatches(q).length, 0); assert.ok(q.storages.every(s => !s.texture.borrowed)); q.renderer.dispose();
});
test('source disposal during buffer upload unwinds the borrow safely before storage release', async () => {
  const r = await rig(), target = new RenderTarget();
  const attribute = {isStorageBufferAttribute: true, array: new Float32Array(1), itemSize: 1, count: 1, onUploadCallback() { target.dispose(); }};
  assert.throws(() => r.renderer.compute(kernel([sampled(target.texture), {isStorageBuffer: true, attribute}])), {code: 'THREE_TARGET_STALE'});
  assert.equal(dispatches(r).length, 0); assert.equal(r.storages[0].disposed, true); assert.equal(r.targets.allocatedBytes, 0); r.renderer.dispose();
});
test('writes through ordinary targets and forged/stale texture associations are refused', async () => {
  const r = await rig(), target = new RenderTarget();
  assert.throws(() => r.renderer.compute(kernel([{...sampled(target.texture), store: true}])), {code: 'THREE_COMPUTE_TARGET_ACCESS'});
  const old = target.texture; target.textures[0] = new RenderTarget().texture; target.texture.renderTarget = target;
  assert.throws(() => r.renderer.compute(kernel([sampled(old)])), {code: 'THREE_TARGET_STALE'});
  assert.equal(r.storages.length, 0); r.renderer.dispose();
});
test('opaque leases are bounded, owner-branded, released idempotently and cannot escape submission', async () => {
  const r = await rig({adapterOptions: {maxBindings: 1}}), target = new RenderTarget();
  const lease = r.adapter.captureBinding(sampled(target.texture));
  assert.throws(() => r.adapter.captureBinding(sampled(target.texture)), {code: 'THREE_COMPUTE_TARGET_BUDGET'});
  assert.throws(() => r.adapter.resolve(lease), {code: 'THREE_COMPUTE_TARGET_BORROW'});
  assert.throws(() => r.adapter.resolve({}), {code: 'THREE_COMPUTE_TARGET_STALE'});
  r.adapter.release(lease); r.adapter.release(lease);
  assert.equal(r.adapter.diagnostics.bindings, 0); assert.throws(() => r.adapter.layout(lease), {code: 'THREE_COMPUTE_TARGET_STALE'});
  const next = r.adapter.captureBinding(sampled(target.texture)); r.adapter.release(next); r.adapter.dispose(); r.renderer.dispose();
});
test('sampler cache is bounded without destroying shared target resources', async () => {
  const r = await rig({adapterOptions: {maxSamplers: 1}}), target = new RenderTarget();
  for (const wrap of [THREE.ClampToEdgeWrapping, THREE.RepeatWrapping, THREE.MirroredRepeatWrapping]) {
    target.texture.wrapS = wrap; const lease = r.adapter.captureBinding(sampler(target.texture));
    r.adapter.withDispatch([lease], () => r.adapter.resolve(lease)); r.adapter.release(lease); await r.adapter.whenIdle();
    assert.equal(r.adapter.diagnostics.samplers, 1);
  }
  r.adapter.dispose(); assert.equal(r.storages[0].disposed, false); r.renderer.dispose();
});
test('loss and disposal interrupt pending completion even when native scopes never settle', async () => {
  for (const action of ['loss', 'dispose']) {
    const r = await rig(), target = new RenderTarget(), never = deferred(); r.scopePromises.push(never.promise);
    const lease = r.adapter.captureBinding(sampled(target.texture)); r.adapter.withDispatch([lease], () => r.adapter.resolve(lease));
    const wait = r.adapter.whenIdle(), rejected = assert.rejects(wait, {code: action === 'loss' ? 'THREE_COMPUTE_TARGET_LOST' : 'THREE_COMPUTE_TARGET_DISPOSED'});
    if (action === 'loss') r.lost.resolve({message: 'removed'}); else r.adapter.dispose();
    await rejected; never.resolve(null); r.renderer.dispose();
  }
});
test('target lease budgets recover after repeated invalid public dispatch capture', async () => {
  const r = await rig(), target = new RenderTarget(), node = kernel([sampled(target.texture)]); node.count = -1;
  for (let i = 0; i < 4200; i++) assert.throws(() => r.renderer.compute(node), {code: 'THREE_COMPUTE_DISPATCH'});
  node.count = 1; r.renderer.compute(node); assert.equal(dispatches(r).length, 1); r.renderer.dispose();
});
