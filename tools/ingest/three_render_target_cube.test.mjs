/** Source cube ownership and recorded GPU boundaries, not pixel equivalence. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {createThreeRenderTargets, inspectThreeRenderTarget, inspectThreeRenderTargetMip} from './three_render_targets.mjs';
import {targetDevice} from './fixtures/target_residency_fixture.mjs';

function fixture(options = {}) {
  const f = targetDevice(); f.device.limits.maxTextureArrayLayers = 256;
  const owner = createThreeRenderTargets(THREE, {getDevice: () => f.device, ...options});
  return {...f, owner};
}
function cube(size = 8) {
  const target = new THREE.CubeRenderTarget(size);
  target.depthTexture = new THREE.CubeDepthTexture(size);
  target.depthTexture.compareFunction = THREE.LessEqualCompare;
  return target;
}
const native = (f, options = {}) => createGpuRenderTarget(f.device,
  {width: 8, height: 8, format: 'rgba8unorm', cube: true, sampleDepth: true, ...options});

test('exact source cube captures without mutation and renders six isolated attachment faces', async () => {
  const f = fixture(), target = cube(), entry = f.owner.capture(target), shape = entry.shape;
  assert.equal(target.texture.renderTarget, null, 'the retained constructor leaves its color owner pointer null');
  assert.equal(target.depth, 1, 'source depth is not the GPU array layer count');
  assert.equal(f.calls.textures.length, 0);
  assert.equal(shape.cube, true); assert.equal(shape.layers, 6); assert.equal(shape.viewDimension, 'cube');
  assert.equal(shape.bytes, 8 * 8 * 6 * (4 + 4)); assert.equal(shape.colorBytes, 8 * 8 * 6 * 4);
  const frames = [];
  for (let face = 0; face < 6; face++) {
    f.owner.render(entry, frame => {
      frames.push(frame);
      for (const view of [frame.colorView, frame.depthView]) {
        assert.equal(view.descriptor.dimension, '2d');
        assert.equal(view.descriptor.baseArrayLayer, face);
        assert.equal(view.descriptor.arrayLayerCount, 1);
        assert.equal(view.texture.size.depthOrArrayLayers, 6);
      }
      const encoder = f.device.createCommandEncoder();
      const pass = encoder.beginRenderPass({colorAttachments: [{view: frame.colorView, loadOp: 'clear', storeOp: 'store'}],
        depthStencilAttachment: {view: frame.depthView, depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1}});
      pass.end(); f.device.queue.submit([encoder.finish()]);
    }, {face});
  }
  assert.equal(new Set(frames).size, 6);
  assert.equal(new Set(frames.map(frame => frame.depthView.texture)).size, 1);
  assert.equal(f.calls.textures.length, 2); assert.equal(f.calls.submissions.length, 6);
  assert.equal(f.owner.allocatedBytes, shape.bytes);
  const source = f.owner.captureNativeTexture(target.depthTexture, {comparison: true});
  assert.equal(source.viewDimension, 'cube'); assert.equal(source.sampleType, 'depth');
  f.owner.withNativeTexture(source, texture => assert.equal(texture, frames[0].depthView.texture));
  assert.equal(f.calls.textures.length, 2, 'native cube depth sampling allocates no 2D replica');
  assert.equal(target.texture.renderTarget, null); assert.equal(target.depth, 1);
  await f.owner.whenIdle(); f.owner.dispose();
  for (const texture of f.calls.textures) assert.equal(texture.destroyed, 1);
});

test('cube destination face and mip rejection happen before native allocation', () => {
  const f = fixture(), entry = f.owner.capture(cube());
  for (const face of [-1, 6, .5, NaN]) {
    assert.throws(() => inspectThreeRenderTargetMip(entry.shape, 0, face), {code: 'THREE_TARGET_FACE'});
    assert.throws(() => f.owner.render(entry, () => assert.fail('invalid face'), {face}), {code: 'THREE_TARGET_FACE'});
  }
  for (const mipLevel of [1, -1, .5])
    assert.throws(() => f.owner.render(entry, () => assert.fail('invalid mip'), {mipLevel}), {code: 'THREE_TARGET_MIP'});
  const ordinary = inspectThreeRenderTarget(new THREE.RenderTarget(8, 8), THREE);
  assert.equal(ordinary.cube, false); assert.equal(ordinary.layers, 1); assert.equal(ordinary.viewDimension, '2d');
  assert.throws(() => inspectThreeRenderTargetMip(ordinary, 0, 1), {code: 'THREE_TARGET_FACE'});
  assert.equal(f.calls.textures.length, 0); f.owner.dispose();
});

test('unsupported source cube profiles are rejected before storage or source descriptor rewrites', () => {
  const f = fixture();
  for (const mutate of [
    target => { target.samples = 4; },
    target => { target.texture.mipmaps = [{}, {}, {}, {}]; },
    target => { target.texture.type = THREE.HalfFloatType; target.texture.colorSpace = THREE.SRGBColorSpace; },
    target => { target.width = 4; },
    target => { target.stencilBuffer = true; },
    target => { target.depthTexture = new THREE.DepthTexture(8, 8); },
    target => { target.depthTexture.format = THREE.DepthStencilFormat; target.depthTexture.type = THREE.UnsignedInt248Type; },
    target => { target.texture.renderTarget = new THREE.RenderTarget(); },
    target => { target.texture.image = target.texture.image.slice(1); },
  ]) {
    const target = cube(); mutate(target);
    assert.throws(() => f.owner.capture(target), error => /^THREE_TARGET_/.test(error.code));
  }
  const ordinary = new THREE.RenderTarget(8, 8); ordinary.depthTexture = new THREE.CubeDepthTexture(8);
  assert.throws(() => f.owner.capture(ordinary), {code: 'THREE_TARGET_UNSUPPORTED'});
  assert.equal(f.calls.textures.length, 0); f.owner.dispose();
});

test('six-layer source cost is enforced before grouped native allocations', () => {
  const f = fixture({maxBytes: 8 * 8 * 6 * 8 - 1}), target = cube(), entry = f.owner.capture(target);
  const source = f.owner.captureNativeTexture(target.depthTexture, {comparison: true});
  assert.throws(() => f.owner.validateNativeTextures([source]), {code: 'THREE_TARGET_BUDGET'});
  assert.throws(() => f.owner.ensure(entry), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(f.calls.textures.length, 0); f.owner.dispose();
  const limited = fixture(); limited.device.limits.maxTextureArrayLayers = 5;
  const captured = limited.owner.captureNativeTexture(cube().depthTexture, {comparison: true});
  assert.throws(() => limited.owner.validateNativeTextures([captured]), {code: 'THREE_TARGET_DEVICE'});
  assert.equal(limited.calls.textures.length, 0); limited.owner.dispose();
});

test('source cube resize refusal preserves accepted storage; public resize invalidates it', () => {
  const f = fixture(), target = cube(), entry = f.owner.capture(target), release = f.owner.retainGeneration(entry);
  f.owner.render(entry, () => {}, {face: 5});
  let ran = false;
  assert.throws(() => f.owner.withNativeShadowResize(target, 16, 16, () => {
    ran = true; target.setSize(16, 16);
  }), {code: 'THREE_TARGET_UNSUPPORTED'});
  assert.equal(ran, false); assert.equal(target.width, 8); assert.equal(entry.alive, true);
  f.owner.render(entry, () => {}, {face: 0});
  target.setSize(16, 16);
  assert.equal(target.texture.image[0].width, 8, 'the pinned source retains its original face descriptors');
  assert.throws(() => f.owner.assertCurrent(entry), {code: 'THREE_TARGET_STALE'});
  assert.throws(() => f.owner.capture(target), {code: 'THREE_TARGET_UNSUPPORTED'});
  release(); f.owner.dispose();
});

test('cube source transfers, color sampling and float depth replicas fail without allocating', () => {
  const f = fixture(), target = cube(), entry = f.owner.capture(target);
  const ordinary = f.owner.capture(new THREE.RenderTarget(8, 8));
  for (const operation of [
    () => f.owner.writePixels(entry, new Uint8Array(8 * 8 * 4)),
    () => f.owner.copyColor(entry, ordinary), () => f.owner.copyColor(ordinary, entry),
    () => f.owner.copyDepth(entry, ordinary), () => f.owner.copyDepth(ordinary, entry),
    () => f.owner.bindingsFor().get(target.depthTexture),
  ]) assert.throws(operation, {code: 'THREE_TARGET_UNSUPPORTED'});
  assert.throws(() => f.owner.captureNativeTexture(target.texture), {code: 'THREE_TARGET_SOURCE'});
  // Even a caller-supplied color pointer must not make this unsupported route
  // fall through to the ordinary 2D binding or transfer implementation.
  target.texture.renderTarget = target;
  assert.throws(() => f.owner.captureNativeTexture(target.texture), {code: 'THREE_TARGET_UNSUPPORTED'});
  assert.throws(() => f.owner.bindingsFor().get(target.texture), {code: 'THREE_TARGET_UNSUPPORTED'});
  assert.equal(f.calls.textures.length, 0); assert.equal(f.calls.submissions.length, 0); f.owner.dispose();
});

test('native cube preflight requires six layers, square faces and the complete six-face budget', () => {
  for (const options of [
    {maxBytes: 8 * 8 * 6 * 8 - 1}, {width: 8, height: 4}, {sampleCount: 4},
    {mipLevelCount: 5}, {copyDestination: true},
  ]) {
    const f = fixture(); assert.throws(() => native(f, options), error => /^GPU_TARGET_/.test(error.code));
    assert.equal(f.calls.textures.length, 0); f.owner.dispose();
  }
  const f = fixture(); f.device.limits.maxTextureArrayLayers = 5;
  assert.throws(() => native(f), {code: 'GPU_TARGET_DEVICE'});
  assert.equal(f.calls.textures.length, 0); f.owner.dispose();
});

test('cube resize charges and retains all six old faces through the last submitted-use fence', async () => {
  const f = fixture(), target = native(f, {maxBytes: (8 * 8 + 4 * 4) * 6 * 8});
  const first = f.calls.textures.slice(); target.withDepthTexture(() => {});
  let finish; const fence = new Promise(resolve => { finish = resolve; });
  f.device.queue.onSubmittedWorkDone = () => fence;
  target.resize(4, 4);
  assert.equal(target.allocatedBytes, (8 * 8 + 4 * 4) * 6 * 8);
  assert.equal(target.diagnostics.retiredTargets, 1);
  first.forEach(texture => assert.equal(texture.destroyed, 0));
  const count = f.calls.textures.length;
  assert.throws(() => target.resize(8, 8), {code: 'GPU_TARGET_BUDGET'});
  assert.equal(f.calls.textures.length, count); assert.equal(target.failed, false);
  finish(); await target.whenIdle();
  first.forEach(texture => assert.equal(texture.destroyed, 1));
  assert.equal(target.allocatedBytes, 4 * 4 * 6 * 8);
  target.resize(8, 8); await target.whenIdle(); target.dispose(); f.owner.dispose();
  f.calls.textures.forEach(texture => assert.equal(texture.destroyed, 1));
});

test('native cube readbacks and transfers never silently operate on face zero', async () => {
  const f = fixture(), target = native(f), ordinary = createGpuRenderTarget(f.device,
    {width: 8, height: 8, format: 'rgba8unorm', sampleDepth: true, copyDestination: true});
  const allocations = f.calls.textures.length;
  for (const operation of [
    () => target.writePixels(new Uint8Array(8 * 8 * 4)),
    () => target.copyFrom(ordinary), () => ordinary.copyFrom(target),
    () => target.copyDepthFrom(ordinary), () => ordinary.copyDepthFrom(target),
  ]) assert.throws(operation, {code: 'GPU_TARGET_UNSUPPORTED'});
  await assert.rejects(target.readPixels(), {code: 'GPU_TARGET_UNSUPPORTED'});
  await assert.rejects(target.readDepthPixels(), {code: 'GPU_TARGET_UNSUPPORTED'});
  assert.throws(() => target.withFrame(() => {}, {face: 6}), {code: 'GPU_TARGET_FACE'});
  assert.throws(() => target.withFrame(() => {}, {mipLevel: 1}), {code: 'GPU_TARGET_MIP'});
  assert.equal(f.calls.textures.length, allocations); assert.equal(f.calls.submissions.length, 0);
  target.dispose(); ordinary.dispose(); f.owner.dispose();
});

test('partial cube face view failure releases both whole native attachments exactly once', () => {
  const f = fixture(), createTexture = f.device.createTexture;
  f.device.createTexture = descriptor => {
    const texture = createTexture(descriptor), createView = texture.createView;
    texture.createView = options => {
      if (options?.baseArrayLayer === 3 && descriptor.format.startsWith('depth')) throw Error('cube face allocation');
      return createView(options);
    };
    return texture;
  };
  assert.throws(() => native(f), /cube face allocation/);
  assert.equal(f.calls.textures.length, 2);
  f.calls.textures.forEach(texture => assert.equal(texture.destroyed, 1));
  assert.equal(f.calls.scopes.length, 0); f.owner.dispose();
});
