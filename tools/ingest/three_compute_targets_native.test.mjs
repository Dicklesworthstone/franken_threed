/** Execute the real native GPU target owner as well as the source target owner
 * and compute adapter. The GPU records descriptors/commands; it does not run
 * shaders. Unused readback/transfer imports throw rather than simulate results. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import {THREE, depthTarget, sampled, sampler, plain} from './three_compute_targets_test_fixture.mjs';
import {createThreeComputeTextures} from './three_compute_textures.mjs';

async function rig(three = THREE) {
  const textures = [], events = [], errors = [];
  const device = {limits: {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256}, features: new Set(), lost: new Promise(() => {}),
    queue: {submit() { events.push('submit'); }, async onSubmittedWorkDone() { events.push('fence'); }},
    pushErrorScope() { events.push('push'); }, popErrorScope() { events.push('pop'); return Promise.resolve(errors.shift() ?? null); },
    createTexture(descriptor) {
      const texture = {descriptor, destroyed: false, createView(view = {}) {
        assert.equal(texture.destroyed, false);
        assert.ok((view.baseMipLevel ?? 0) + (view.mipLevelCount ?? 1) <= descriptor.mipLevelCount);
        assert.ok((view.baseArrayLayer ?? 0) + (view.arrayLayerCount ?? 1) <= (descriptor.size.depthOrArrayLayers ?? 1));
        return {texture, descriptor: plain(view)};
      }, destroy() { texture.destroyed = true; }};
      textures.push(texture); return texture;
    },
    createSampler(descriptor) { return {descriptor}; }, createShaderModule(descriptor) { return {descriptor}; },
    createBindGroupLayout(descriptor) { return descriptor; }, createPipelineLayout(descriptor) { return descriptor; },
    createRenderPipeline(descriptor) { return {descriptor, getBindGroupLayout() { return {}; }}; },
    createBindGroup(descriptor) { return descriptor; },
    createCommandEncoder() { return {beginRenderPass() {
      events.push('mipPass'); return {setPipeline() {}, setBindGroup() {}, draw() {}, end() {}};
    }, finish() { return {}; }}; },
  };
  const context = vm.createContext({console, AbortController, Map, Set, WeakMap, WeakSet});
  const missing = () => { throw new Error('This test must not execute a readback or transfer seam'); };
  const synthetic = exports => new vm.SyntheticModule(Object.keys(exports), function () {
    for (const [key, value] of Object.entries(exports)) this.setExport(key, value);
  }, {context});
  const modules = new Map([
    ['./gpu_target_readback.mjs', synthetic({readGpuTargetPixels: missing, readGpuTargetDepth: missing})],
    ['./gpu_target_transfer.mjs', synthetic({inspectGpuTargetWrite: missing, gpuTargetWriteData: missing, inspectGpuTargetCopy: missing})],
    ['./gpu_depth_transfer.mjs', synthetic({inspectGpuDepthCopy: missing, submitGpuDepthCopy: missing})],
  ]);
  for (const name of ['gpu_render_target', 'three_render_targets', 'three_compute_targets'])
    modules.set(`./${name}.mjs`, new vm.SourceTextModule(await readFile(new URL(`./${name}.mjs`, import.meta.url), 'utf8'), {context}));
  const link = name => { assert.ok(modules.has(name), name); return modules.get(name); };
  for (const module of modules.values()) if (module.status === 'unlinked') await module.link(link);
  for (const module of modules.values()) if (module.status !== 'evaluated') await module.evaluate();
  const create = options => modules.get('./gpu_render_target.mjs').namespace.createGpuRenderTarget(device, options);
  const targets = modules.get('./three_render_targets.mjs').namespace.createThreeRenderTargets(three, {getDevice: () => device});
  const adapter = modules.get('./three_compute_targets.mjs').namespace.createThreeComputeTargets({targets, device});
  return {create, targets, adapter, textures, events, errors, device, dispose() { adapter.dispose(); targets.dispose(); }};
}

test('actual native owner exposes original color and depth together without replica allocations', async () => {
  const r = await rig(), target = depthTarget(THREE.UnsignedInt248Type, THREE.DepthStencilFormat);
  const color = r.adapter.captureBinding(sampled(target.texture)), depth = r.adapter.captureBinding(sampled(target.depthTexture));
  r.adapter.withDispatch([depth, color], () => {
    assert.equal(r.adapter.resolve(color).texture, r.textures[0]); assert.equal(r.adapter.resolve(depth).texture, r.textures[1]);
    assert.equal(r.textures[1].descriptor.format, 'depth24plus-stencil8');
    assert.equal(r.adapter.resolve(depth).descriptor.aspect, 'depth-only');
    assert.equal(r.targets.capture(target).busy, true); r.device.queue.submit([]);
  });
  assert.equal(r.textures.length, 2); assert.equal(r.events.filter(e => e === 'mipPass').length, 0);
  assert.equal(r.targets.capture(target).storage.diagnostics.mipGenerations, 0);
  await r.adapter.whenIdle(); r.dispose(); assert.ok(r.textures.every(t => t.destroyed));
});
test('actual color mips are generated before the joint consumer and not by depth-only reads', async () => {
  const r = await rig(), target = depthTarget(); target.texture.generateMipmaps = true;
  const depth = r.adapter.captureBinding(sampled(target.depthTexture)), color = r.adapter.captureBinding(sampled(target.texture));
  r.adapter.withDispatch([depth], () => r.adapter.resolve(depth)); assert.equal(r.events.filter(e => e === 'mipPass').length, 0);
  r.adapter.withDispatch([color, depth], () => {
    assert.equal(r.events.filter(e => e === 'mipPass').length, 2);
    assert.equal(r.adapter.resolve(color).descriptor.mipLevelCount, 3);
    assert.equal(r.adapter.resolve(depth).descriptor.mipLevelCount, 1);
  });
  r.adapter.withDispatch([depth, color], () => {});
  assert.equal(r.targets.capture(target).storage.diagnostics.mipGenerations, 1);
  r.targets.render(r.targets.capture(target), () => {});
  r.adapter.withDispatch([depth], () => {});
  assert.equal(r.targets.capture(target).storage.diagnostics.mipGenerations, 1);
  r.adapter.withDispatch([depth, color], () => {});
  assert.equal(r.targets.capture(target).storage.diagnostics.mipGenerations, 2);
  await r.adapter.whenIdle(); r.dispose();
});
test('joint borrowing preserves native exclusivity across every resize/render/sample/dispose entry point', async () => {
  const r = await rig(), owner = r.create({sampleDepth: true});
  owner.withSampledTextures((color, depth) => {
    assert.notEqual(color, depth);
    for (const operation of [() => owner.resize(8, 8), () => owner.withFrame(() => {}), () => owner.withTexture(() => {}),
      () => owner.withDepthTexture(() => {}), () => owner.withSampledTextures(() => {}), () => owner.dispose()])
      assert.throws(operation, {code: 'GPU_TARGET_REENTRANT'});
  });
  owner.withTexture(() => {}); owner.dispose(); r.dispose();
});
test('a joint borrow cannot cross an await and remains usable after callback failure', async () => {
  const r = await rig(), owner = r.create({sampleDepth: true});
  assert.throws(() => owner.withSampledTextures(async () => {}), {code: 'GPU_TARGET_ASYNC_USE'});
  const error = new Error('consumer failed'); assert.throws(() => owner.withSampledTextures(() => { throw error; }), e => e === error);
  let reached = false; owner.withSampledTextures(() => { reached = true; }); assert.equal(reached, true);
  owner.dispose(); r.dispose();
});
test('joint native sampling requires explicit depth usage and validates callbacks even while suspended', async () => {
  const r = await rig(), privateDepth = r.create({});
  assert.throws(() => privateDepth.withSampledTextures(() => {}), {code: 'GPU_TARGET_DEPTH'});
  privateDepth.dispose();
  const suspended = r.create({width: 0, height: 0, sampleDepth: true});
  assert.throws(() => suspended.withSampledTextures(null), {code: 'GPU_TARGET_CALLBACK'});
  assert.equal(suspended.withSampledTextures(() => assert.fail('suspended callback')), false);
  suspended.dispose(); r.dispose();
});
test('native four-sample depth remains multisampled while color is resolved, without depth conversion', async () => {
  const r = await rig(), owner = r.create({sampleDepth: true, sampleCount: 4});
  owner.withSampledTextures((color, depth) => {
    assert.equal(color.descriptor.sampleCount, 1); assert.equal(depth.descriptor.sampleCount, 4);
    assert.equal(color.descriptor.usage & 4, 4); assert.equal(depth.descriptor.usage & 4, 4);
  });
  assert.equal(r.textures.length, 3); owner.dispose(); r.dispose();
});
test('source retirement during a real joint native borrow releases both attachments only after unwinding', async () => {
  const r = await rig(), target = depthTarget(), color = r.adapter.captureBinding(sampled(target.texture)), depth = r.adapter.captureBinding(sampled(target.depthTexture));
  assert.throws(() => r.adapter.withDispatch([color, depth], () => {
    target.depthTexture.dispose(); assert.ok(r.textures.every(t => !t.destroyed));
  }), {code: 'THREE_TARGET_STALE'});
  assert.ok(r.textures.every(t => t.destroyed)); assert.equal(r.targets.allocatedBytes, 0); r.dispose();
});
test('actual allocation validation is included in dependent compute completion', async () => {
  const r = await rig(), target = depthTarget(), depth = r.adapter.captureBinding(sampled(target.depthTexture));
  r.errors.push({message: 'depth allocation rejected'});
  r.adapter.withDispatch([depth], () => r.adapter.resolve(depth));
  await assert.rejects(r.adapter.whenIdle(), {code: 'GPU_TARGET_GPU'});
  assert.ok(r.textures.every(t => t.destroyed)); r.dispose();
});
test('ordinary native depth uses non-filtering nearest samplers even with source linear filters', async () => {
  const r = await rig(), target = depthTarget(); target.depthTexture.minFilter = target.depthTexture.magFilter = THREE.LinearFilter;
  const lease = r.adapter.captureBinding(sampler(target.depthTexture));
  assert.deepEqual(plain(r.adapter.layout(lease)), {sampler: {type: 'non-filtering'}});
  r.adapter.withDispatch([lease], () => {
    const descriptor = r.adapter.resolve(lease).descriptor;
    assert.equal(descriptor.minFilter, 'nearest'); assert.equal(descriptor.magFilter, 'nearest'); assert.equal(descriptor.compare, undefined);
    assert.equal(descriptor.mipmapFilter, 'nearest');
  });
  assert.equal(r.device.features.size, 0); await r.adapter.whenIdle(); r.dispose();
});
test('comparison and ordinary depth samplers share the real attachment and retain independent captured state', async () => {
  const r = await rig(), target = depthTarget(), depth = target.depthTexture;
  depth.compareFunction = THREE.LessEqualCompare;
  depth.minFilter = depth.magFilter = THREE.LinearFilter;
  const read = r.adapter.captureBinding(sampled(depth));
  const first = r.adapter.captureBinding({...sampler(depth), comparison: true});
  const ordinary = r.adapter.captureBinding(sampler(depth));
  depth.compareFunction = THREE.GreaterCompare;
  depth.minFilter = depth.magFilter = THREE.NearestFilter;
  const second = r.adapter.captureBinding({...sampler(depth), comparison: true});
  assert.deepEqual(plain(r.adapter.layout(read)), {texture: {sampleType: 'depth', viewDimension: '2d', multisampled: false}});
  assert.deepEqual(plain(r.adapter.layout(first)), {sampler: {type: 'comparison'}});
  assert.deepEqual(plain(r.adapter.layout(ordinary)), {sampler: {type: 'non-filtering'}});
  depth.compareFunction = null;
  r.adapter.withDispatch([read, first, ordinary, second], () => {
    const earlier = r.adapter.resolve(first).descriptor, later = r.adapter.resolve(second).descriptor;
    assert.equal(earlier.compare, 'less-equal'); assert.equal(earlier.minFilter, 'linear'); assert.equal(earlier.magFilter, 'linear');
    assert.equal(later.compare, 'greater'); assert.equal(later.minFilter, 'nearest');
    assert.equal(r.adapter.resolve(ordinary).descriptor.compare, undefined);
    assert.equal(r.adapter.resolve(ordinary).descriptor.magFilter, 'nearest');
    assert.equal(r.adapter.resolve(read).texture, r.textures[1]);
    assert.equal(r.adapter.resolve(read).descriptor.aspect, 'depth-only');
    assert.equal(r.targets.capture(target).busy, true);
  });
  assert.equal(r.textures.length, 2); assert.equal(r.targets.capture(target).depthSample, null);
  assert.equal(r.events.filter(e => e === 'mipPass').length, 0); assert.equal(r.device.features.size, 0);
  await r.adapter.whenIdle(); r.dispose();
});
test('native depth supports all pinned source comparison functions', async () => {
  const r = await rig(), target = depthTarget();
  const expected = [[THREE.NeverCompare, 'never'], [THREE.LessCompare, 'less'], [THREE.EqualCompare, 'equal'],
    [THREE.LessEqualCompare, 'less-equal'], [THREE.GreaterCompare, 'greater'], [THREE.NotEqualCompare, 'not-equal'],
    [THREE.GreaterEqualCompare, 'greater-equal'], [THREE.AlwaysCompare, 'always']];
  const leases = expected.map(([compareFunction]) => {
    target.depthTexture.compareFunction = compareFunction;
    return r.adapter.captureBinding({...sampler(target.depthTexture), comparison: true});
  });
  r.adapter.withDispatch(leases, () => assert.deepEqual(leases.map(lease => r.adapter.resolve(lease).descriptor.compare), expected.map(([, value]) => value)));
  assert.equal(r.textures.length, 2); await r.adapter.whenIdle(); r.dispose();
});
test('comparison samplers require a valid comparison depth texture before native allocation', async () => {
  const r = await rig(), target = depthTarget();
  for (const texture of [target.texture, target.depthTexture])
    assert.throws(() => r.adapter.captureBinding({...sampler(texture), comparison: true}), {code: 'THREE_TARGET_SAMPLER'});
  target.depthTexture.compareFunction = 999;
  assert.throws(() => r.adapter.captureBinding({...sampler(target.depthTexture), comparison: true}), {code: 'THREE_TARGET_SAMPLER'});
  assert.equal(r.textures.length, 0); r.dispose();
});
test('comparison leases preserve resize invalidation before any native submission', async () => {
  const r = await rig(), target = depthTarget(); target.depthTexture.compareFunction = THREE.LessEqualCompare;
  const read = r.adapter.captureBinding(sampled(target.depthTexture));
  const compare = r.adapter.captureBinding({...sampler(target.depthTexture), comparison: true});
  target.setSize(8, 8);
  assert.throws(() => r.adapter.withDispatch([read, compare], () => assert.fail('stale comparison consumer')), {code: 'THREE_TARGET_STALE'});
  assert.equal(r.textures.length, 0); assert.equal(r.events.includes('submit'), false); r.dispose();
});

test('source shadow resize leases distinguish generations and retire only after submitted readers finish', async () => {
  const r = await rig(), target = depthTarget(); target.depthTexture.compareFunction = THREE.LessEqualCompare;
  let complete;
  const completion = new Promise(resolve => { complete = resolve; });
  r.device.queue.onSubmittedWorkDone = () => completion;
  const previous = r.adapter.captureBinding(sampled(target.depthTexture));
  r.targets.withNativeShadowResize(target, 8, 8, () => target.setSize(8, 8));
  const current = r.adapter.captureBinding(sampled(target.depthTexture));
  assert.equal(r.targets.size, 2);
  r.adapter.withDispatch([previous, current], () => {
    const oldView = r.adapter.resolve(previous), newView = r.adapter.resolve(current);
    assert.notEqual(oldView, newView);
    assert.equal(oldView.texture.descriptor.size.width, 4);
    assert.equal(newView.texture.descriptor.size.width, 8);
  });
  r.adapter.release(previous);
  assert.ok(r.textures.every(texture => !texture.destroyed), 'pending GPU validation keeps the retired generation alive');
  assert.equal(r.targets.allocatedBytes, (4 * 4 + 8 * 8) * 8);
  complete(); await r.adapter.whenIdle(); await r.targets.whenIdle();
  assert.ok(r.textures.slice(0, 2).every(texture => texture.destroyed));
  assert.ok(r.textures.slice(2).every(texture => !texture.destroyed));
  assert.equal(r.targets.size, 1); assert.equal(r.targets.allocatedBytes, 8 * 8 * 8);
  r.adapter.release(current); r.dispose();
});

test('pinned cube depth borrows all six native layers with independent comparison sampler histories', async () => {
  const three = await import('../../upstream/three.js/build/three.webgpu.js');
  const r = await rig(three), target = new three.CubeRenderTarget(8);
  target.depthTexture = new three.CubeDepthTexture(8);
  const depth = target.depthTexture;
  assert.equal(depth.isDepthTexture, true); assert.equal(depth.isCubeTexture, true); assert.equal(depth.isCubeDepthTexture, true);
  depth.compareFunction = three.LessEqualCompare;
  depth.minFilter = depth.magFilter = three.LinearFilter;
  const read = r.adapter.captureBinding(sampled(depth));
  const first = r.adapter.captureBinding({...sampler(depth), comparison: true});
  const ordinary = r.adapter.captureBinding(sampler(depth));
  depth.compareFunction = three.GreaterCompare;
  depth.minFilter = depth.magFilter = three.NearestFilter;
  const second = r.adapter.captureBinding({...sampler(depth), comparison: true});
  assert.deepEqual(plain(r.adapter.layout(read)), {texture: {sampleType: 'depth', viewDimension: 'cube', multisampled: false}});
  assert.deepEqual(plain(r.adapter.layout(first)), {sampler: {type: 'comparison'}});
  assert.deepEqual(plain(r.adapter.layout(ordinary)), {sampler: {type: 'non-filtering'}});
  assert.deepEqual(plain(r.adapter.sampleData(depth)), {samples: 1, primarySamples: 1, isMSAA: false});
  depth.compareFunction = null;
  r.adapter.withDispatch([read, first, ordinary, second], () => {
    const view = r.adapter.resolve(read), earlier = r.adapter.resolve(first).descriptor, later = r.adapter.resolve(second).descriptor;
    assert.equal(view.texture, r.textures[1]);
    assert.deepEqual(view.descriptor, {dimension: 'cube', baseMipLevel: 0, mipLevelCount: 1,
      baseArrayLayer: 0, arrayLayerCount: 6, aspect: 'depth-only'});
    assert.equal(r.adapter.resolve(read), view, 'all native consumers reuse the same cube view');
    assert.equal(view.texture.descriptor.size.depthOrArrayLayers, 6);
    assert.equal(view.texture.descriptor.format, 'depth24plus');
    assert.equal(view.texture.descriptor.sampleCount, 1);
    assert.equal(earlier.compare, 'less-equal'); assert.equal(earlier.minFilter, 'linear'); assert.equal(earlier.magFilter, 'linear');
    assert.equal(later.compare, 'greater'); assert.equal(later.minFilter, 'nearest');
    assert.equal(r.adapter.resolve(ordinary).descriptor.compare, undefined);
    assert.equal(r.adapter.resolve(ordinary).descriptor.minFilter, 'nearest');
    assert.equal(r.targets.capture(target).busy, true);
    r.device.queue.submit([]);
  });
  const entry = r.targets.capture(target);
  assert.equal(r.textures.length, 2); assert.equal(entry.depthSample, null);
  assert.equal(entry.shape.layers, 6); assert.equal(r.targets.allocatedBytes, 8 * 8 * 6 * 8);
  assert.equal(r.events.filter(event => event === 'mipPass').length, 0);
  assert.equal(r.device.features.size, 0);
  await r.adapter.whenIdle(); r.dispose();
});

test('shared native texture inputs route pinned cube depth to target residency without CPU snapshots or uploads', async () => {
  const three = await import('../../upstream/three.js/build/three.webgpu.js');
  const r = await rig(three), target = new three.CubeRenderTarget(4);
  target.depthTexture = new three.CubeDepthTexture(4);
  target.depthTexture.compareFunction = three.LessEqualCompare;
  const source = target.depthTexture.source, images = target.depthTexture.image;
  const inputs = createThreeComputeTextures({
    storage: {validateDispatch(values) { assert.deepEqual(values, []); }, async whenIdle() {}},
    poolFactory() { assert.fail('Target-owned cube depth must not reach the CPU texture uploader'); },
    targetFactory: () => r.adapter,
  });
  const binding = sampled(target.depthTexture), comparison = {...sampler(target.depthTexture), comparison: true};
  assert.deepEqual(plain(inputs.layoutOf(binding)), {texture: {sampleType: 'depth', viewDimension: 'cube', multisampled: false}});
  const read = inputs.captureBinding(binding), compare = inputs.captureBinding(comparison), leases = [read, compare];
  assert.equal(inputs.preparation(leases), null); inputs.validateDispatch(leases);
  inputs.withDispatch(leases, () => {
    const view = inputs.resolve(read);
    assert.equal(view.texture, r.textures[1]); assert.equal(view.descriptor.dimension, 'cube');
    assert.equal(inputs.resolve(compare).descriptor.compare, 'less-equal');
    r.device.queue.submit([]);
  });
  assert.equal(target.depthTexture.source, source); assert.equal(target.depthTexture.image, images);
  assert.deepEqual(inputs.diagnostics, {textures: 0, versions: 0, snapshotBytes: 0});
  assert.equal(r.textures.length, 2);
  await inputs.whenIdle(); for (const lease of leases) inputs.release(lease);
  assert.equal(r.adapter.diagnostics.bindings, 0);
  inputs.dispose(); r.dispose();
});

for (const retire of ['dispose', 'resize']) test(`pinned cube depth ${retire} invalidates queued cube views and comparison samplers before submission`, async () => {
  const three = await import('../../upstream/three.js/build/three.webgpu.js');
  const r = await rig(three), target = new three.CubeRenderTarget(4);
  target.depthTexture = new three.CubeDepthTexture(4);
  target.depthTexture.compareFunction = three.LessEqualCompare;
  const read = r.adapter.captureBinding(sampled(target.depthTexture));
  const compare = r.adapter.captureBinding({...sampler(target.depthTexture), comparison: true});
  if (retire === 'dispose') target.dispose(); else target.setSize(8, 8);
  assert.throws(() => r.adapter.withDispatch([read, compare], () => assert.fail('stale cube shadow consumer')), {code: 'THREE_TARGET_STALE'});
  assert.equal(r.textures.length, 0); assert.equal(r.events.includes('submit'), false);
  r.dispose();
});
