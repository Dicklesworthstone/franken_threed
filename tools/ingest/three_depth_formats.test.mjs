/** Pinned source depth formats through target, sampling, copy and readback owners.
 * Recording GPU checks preserve ownership and byte/layout contracts; they do not
 * emulate depth quantization or establish browser pixel equivalence. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {getFormat} from '../../upstream/three.js/src/renderers/webgpu/utils/WebGPUTextureUtils.js';
import {createThreeDepthTextures, threeDepthTextureFormat} from './three_depth_textures.mjs';
import {createThreeRenderTargets, inspectThreeRenderTarget} from './three_render_targets.mjs';
import {createThreeComputeTargets} from './three_compute_targets.mjs';
import {depthDevice, deferred} from './tests/depth_transfer_test_fixture.mjs';

const profiles = [
  {name: 'depth16unorm', type: T.UnsignedShortType, format: T.DepthFormat, bytes: 2},
  {name: 'depth32float-stencil8', type: T.FloatType, format: T.DepthStencilFormat, bytes: 8},
];
const sampled = texture => ({texture, isSampledTexture: true});
function depth(profile, width = 4, height = 3) {
  const value = new T.DepthTexture(width, height, profile.type); value.format = profile.format; return value;
}
function target(profile, samples = 0, width = 4, height = 3, options = {}) {
  return new T.RenderTarget(width, height, {samples, depthTexture: depth(profile, width, height),
    stencilBuffer: profile.format === T.DepthStencilFormat, ...options});
}
function targets(t, options = {}) {
  const f = depthDevice(), owner = createThreeRenderTargets(T, {getDevice: () => f.device, ...options});
  t.after(() => owner.dispose()); return {...f, owner};
}

test('shared depth format resolver matches all five pinned source mappings', () => {
  const device = {features: new Set(['depth32float-stencil8'])};
  for (const profile of [...profiles,
    {name: 'depth24plus', type: T.UnsignedIntType, format: T.DepthFormat},
    {name: 'depth32float', type: T.FloatType, format: T.DepthFormat},
    {name: 'depth24plus-stencil8', type: T.UnsignedInt248Type, format: T.DepthStencilFormat}]) {
    const texture = depth(profile), image = texture.image, source = texture.source;
    assert.equal(getFormat(texture, device), profile.name);
    assert.equal(threeDepthTextureFormat(texture, T), profile.name);
    assert.equal(texture.image, image); assert.equal(texture.source, source); assert.equal(texture.version, 0);
    texture.dispose();
  }
  for (const [type, format] of [[T.UnsignedByteType, T.DepthFormat], [T.UnsignedShortType, T.DepthStencilFormat],
    [T.UnsignedInt248Type, T.DepthFormat], [T.UnsignedIntType, T.DepthStencilFormat]])
    assert.equal(threeDepthTextureFormat({type, format}, T), null);
});

for (const profile of profiles) for (const samples of [0, 4])
  test(`${profile.name}: target rendering and native depth bindings share the exact ${samples || 1}-sample allocation`, async t => {
    const f = targets(t), source = target(profile, samples), shape = inspectThreeRenderTarget(source, T);
    const inputs = createThreeComputeTargets({targets: f.owner, device: f.device});
    t.after(() => { inputs.dispose(); source.dispose(); });
    assert.equal(shape.depthFormat, getFormat(source.depthTexture, f.device));
    const expectedBytes = 12 * (4 * (samples === 4 ? 5 : 1) + profile.bytes * (samples || 1));
    assert.equal(shape.bytes, expectedBytes);
    const entry = f.owner.ensure(f.owner.capture(source)); let native;
    f.owner.render(entry, frame => { native = frame.depthView.texture; native.data.fill(.375); });
    const token = inputs.captureBinding(sampled(source.depthTexture));
    assert.deepEqual(inputs.layout(token), {texture: {sampleType: 'depth', viewDimension: '2d', multisampled: samples === 4}});
    inputs.withDispatch([token], () => {
      const view = inputs.resolve(token);
      assert.equal(view.texture, native); assert.equal(view.options.aspect, 'depth-only');
      assert.equal(native.format, profile.name); assert.equal(native.sampleCount, samples || 1);
      assert.ok(native.data.every(value => value === .375)); f.device.queue.submit([]);
    });
    inputs.release(token); await inputs.whenIdle();
    assert.equal(f.owner.allocatedBytes, expectedBytes); assert.equal(entry.storage.allocatedBytes, expectedBytes);
    assert.equal(f.textures.filter(texture => texture.format === profile.name).length, 1);
    assert.equal(entry.depthSample, null, 'native consumers require no float replica');
  });

for (const profile of profiles)
  test(`${profile.name}: shared depth retains one byte charge and owner disposal invalidates queued borrowers`, async t => {
    const f = targets(t), a = target(profile), b = new T.RenderTarget(4, 3, {depthTexture: a.depthTexture});
    const inputs = createThreeComputeTargets({targets: f.owner, device: f.device});
    t.after(() => { inputs.dispose(); a.dispose(); b.dispose(); });
    const first = f.owner.ensure(f.owner.capture(a)), second = f.owner.ensure(f.owner.capture(b)); let native;
    f.owner.render(first, frame => { native = frame.depthView.texture; });
    f.owner.render(second, frame => assert.equal(frame.depthView.texture, native));
    assert.equal(f.owner.allocatedBytes, 12 * (8 + profile.bytes));
    const token = inputs.captureBinding(sampled(a.depthTexture));
    a.dispose(); assert.throws(() => inputs.validateDispatch([token]), {code: 'THREE_TARGET_STALE'});
    inputs.release(token); await f.owner.whenIdle();
    assert.equal(f.owner.allocatedBytes, 0); assert.equal(native.destroyed, true);
  });

test('depth16 target admission uses two depth bytes per pixel before allocation', t => {
  const f = targets(t, {maxBytes: 72}), source = target(profiles[0]); t.after(() => source.dispose());
  const entry = f.owner.ensure(f.owner.capture(source)); assert.equal(f.owner.allocatedBytes, 72);
  assert.equal(entry.storage.allocatedBytes, 72);
  const g = targets(t, {maxBytes: 71}); assert.throws(() => g.owner.ensure(g.owner.capture(source)), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(g.textures.length, 0); assert.equal(g.owner.allocatedBytes, 0);
});

test('float depth/stencil requires its enabled device feature before target or detached allocation', t => {
  const f = targets(t), source = target(profiles[1]); f.device.features.clear(); t.after(() => source.dispose());
  assert.throws(() => f.owner.ensure(f.owner.capture(source)), {code: 'GPU_TARGET_FEATURE'});
  const owner = createThreeDepthTextures({three: T, device: f.device}); t.after(() => owner.dispose());
  const texture = depth(profiles[1]); t.after(() => texture.dispose());
  assert.throws(() => owner.captureBinding(sampled(texture)), {code: 'THREE_DEPTH_TEXTURE_FEATURE'});
  assert.equal(f.textures.length, 0); assert.equal(owner.diagnostics.bytes, 0); assert.equal(f.owner.allocatedBytes, 0);
});

for (const profile of profiles) for (const samples of [1, 4])
  test(`${profile.name}: detached ${samples}-sample framebuffer copies retain all aspects and sample depth only`, async t => {
    const f = depthDevice(), owner = createThreeDepthTextures({three: T, device: f.device, getSamples: () => samples});
    const texture = depth(profile), source = target(profile, samples), targetOwner = createThreeRenderTargets(T, {getDevice: () => f.device});
    t.after(() => { owner.dispose(); targetOwner.dispose(); texture.dispose(); source.dispose(); });
    const entry = targetOwner.ensure(targetOwner.capture(source));
    const copy = owner.captureFramebuffer(texture), read = owner.captureBinding(sampled(texture)), commands = [];
    assert.equal(owner.diagnostics.bytes, 12 * profile.bytes * samples);
    targetOwner.withFramebufferTexture(entry, native => owner.withDispatch([copy, read], () => {
      const view = owner.resolve(read); assert.equal(view.options.aspect, 'depth-only');
      assert.equal(view.texture.format, getFormat(texture, f.device)); assert.equal(view.texture.sampleCount, samples);
      owner.encodeFramebufferCopy(copy, {copyTextureToTexture(...args) { commands.push(args); }}, native);
      assert.equal(commands[0][0].aspect, undefined); assert.equal(commands[0][1].aspect, undefined);
      assert.equal(commands[0][1].texture, view.texture);
      assert.deepEqual(commands[0][2], {width: 4, height: 3, depthOrArrayLayers: 1}); f.device.queue.submit([]);
    }), {depth: true});
    owner.release(copy); owner.release(read); await owner.whenIdle();
    assert.equal(owner.diagnostics.bindings, 0); assert.equal(entry.references, 0);
  });

for (const profile of profiles)
  test(`${profile.name}: depth-only transfer and Float32 readback preserve unrelated color/stencil`, async t => {
    const f = targets(t), source = target(profile), destination = target(profile);
    t.after(() => { source.dispose(); destination.dispose(); });
    const a = f.owner.ensure(f.owner.capture(source)), b = f.owner.ensure(f.owner.capture(destination));
    let color, stencil;
    f.owner.render(a, frame => frame.depthView.texture.data.fill(.625));
    f.owner.render(b, frame => {
      color = frame.colorView.texture; color.data.fill(.25);
      stencil = frame.depthView.texture.stencil; stencil.fill(91); frame.depthView.texture.data.fill(.125);
    });
    await f.owner.copyDepth(a, b, {sourceX: 1, sourceY: 1, x: 0, y: 0, width: 2, height: 2});
    const values = await b.storage.readDepthPixels({flipY: false});
    assert.ok(values instanceof Float32Array);
    assert.deepEqual([...values], [.625, .625, .125, .125, .625, .625, .125, .125, .125, .125, .125, .125]);
    assert.ok(color.data.every(value => value === .25)); assert.ok(stencil.every(value => value === 91));
    assert.equal(f.pipelines[0].desc.depthStencil.format, profile.name);
  });

test('queued detached depth precision changes retain the accepted format and release its exact byte charge', async t => {
  const f = depthDevice(), owner = createThreeDepthTextures({three: T, device: f.device});
  const texture = depth(profiles[0]), first = owner.captureBinding(sampled(texture));
  t.after(() => { owner.dispose(); texture.dispose(); });
  texture.type = T.FloatType; texture.format = T.DepthStencilFormat; texture.needsUpdate = true;
  const second = owner.captureBinding(sampled(texture));
  assert.equal(owner.diagnostics.bytes, 12 * 10);
  const fence = deferred(); f.control.fence = fence.promise; let old;
  owner.withDispatch([first], () => { old = owner.resolve(first).texture; assert.equal(old.format, 'depth16unorm'); f.device.queue.submit([]); });
  owner.withDispatch([second], () => { assert.equal(owner.resolve(second).texture.format, 'depth32float-stencil8'); f.device.queue.submit([]); });
  owner.release(first); assert.equal(old.destroyed, false); assert.equal(owner.diagnostics.bytes, 120);
  fence.resolve(); await owner.whenIdle(); assert.equal(old.destroyed, true); assert.equal(owner.diagnostics.bytes, 96);
  texture.dispose(); owner.release(second); await owner.whenIdle(); assert.equal(owner.diagnostics.bytes, 0);
});
