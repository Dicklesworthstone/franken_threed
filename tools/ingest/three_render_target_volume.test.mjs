/** Pinned source descriptors and native owner recording, not GPU pixel parity. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeRenderTargets, inspectThreeRenderTarget, inspectThreeRenderTargetMip} from './three_render_targets.mjs';
import {targetDevice} from './fixtures/target_residency_fixture.mjs';

function fixture(options = {}) {
  const f = targetDevice();
  Object.assign(f.device.limits, {maxTextureDimension3D: 256, maxTextureArrayLayers: 256,
    maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  return {...f, owner: createThreeRenderTargets(THREE, {getDevice: () => f.device, ...options})};
}
const target = options => new THREE.RenderTarget3D(8, 4, 5, options);

test('source volume MRT retains native 3D colors and ordinary shared depth', async t => {
  const f = fixture(), source = target({count: 2}); t.after(() => f.owner.dispose());
  source.textures[1].type = THREE.HalfFloatType; source.textures[1].format = THREE.RGFormat;
  const entry = f.owner.capture(source), {shape} = entry;
  assert.equal(f.calls.textures.length, 0);
  assert.deepEqual([shape.volume, shape.array, shape.depth, shape.layers, shape.depthLayers], [true, false, 5, 5, 1]);
  assert.equal(shape.viewDimension, '3d'); assert.equal(shape.depthViewDimension, '2d');
  assert.deepEqual(shape.colorFormats, ['rgba8unorm', 'rg16float']); assert.equal(shape.bytes, 8 * 4 * (5 * 8 + 4));
  const frames = [];
  for (let face = 0; face < 5; face++) f.owner.render(entry, frame => {
    frames.push(frame); assert.equal(frame.depthSlice, face);
    assert.deepEqual(frame.colorViews.map(view => view.descriptor.dimension), ['3d', '3d']);
  }, {face});
  assert.ok(frames.every(frame => frame.depthView === frames[0].depthView));
  const inputs = source.textures.map(texture => f.owner.captureNativeTexture(texture));
  assert.ok(inputs.every(input => input.viewDimension === '3d'));
  f.owner.withNativeTextures(inputs, textures => {
    assert.equal(textures.size, 2);
    for (const texture of source.textures) assert.equal(textures.get(texture).size.depthOrArrayLayers, 5);
  });
  assert.equal(f.calls.textures.length, 3); assert.equal(f.owner.allocatedBytes, shape.bytes);
  assert.throws(() => f.owner.bindingsFor().get(source.texture), {code: 'THREE_TARGET_UNSUPPORTED'});
  await f.owner.whenIdle();
});

for (const authored of [false, true])
test(`volume depth has independently selected array layers (authored=${authored})`, async t => {
  const f = fixture(), source = target({useArrayDepthTexture: !authored}); t.after(() => f.owner.dispose());
  source.depthTexture = new THREE.DepthTexture(8, 4, THREE.FloatType);
  if (authored) source.depthTexture.image.depth = 5;
  source.depthTexture.compareFunction = THREE.LessEqualCompare;
  const entry = f.owner.capture(source), input = f.owner.captureNativeTexture(source.depthTexture, {comparison: true});
  assert.equal(entry.shape.depthLayers, 5); assert.equal(input.viewDimension, '2d-array'); assert.equal(input.sampleType, 'depth');
  assert.equal(input.sampler.compare, 'less-equal');
  f.owner.render(entry, frame => {
    assert.equal(frame.depthSlice, 4); assert.equal(frame.depthView.descriptor.baseArrayLayer, 4);
    assert.equal(frame.depthView.descriptor.dimension, '2d');
  }, {face: 4});
  f.owner.withNativeTexture(input, texture => assert.equal(texture.dimension, '2d'));
  assert.equal(f.owner.allocatedBytes, 8 * 4 * 5 * 8); await f.owner.whenIdle();
});

test('legacy WebGL3D target resolves registered ownership without rewriting its null source pointer', t => {
  const f = fixture(), source = new THREE.WebGL3DRenderTarget(8, 4, 5); t.after(() => f.owner.dispose());
  assert.equal(source.texture.renderTarget, null); assert.equal(f.owner.getTextureTarget(source.texture), null);
  const entry = f.owner.capture(source), input = f.owner.captureNativeTexture(source.texture);
  assert.equal(input.entry, entry); assert.equal(input.viewDimension, '3d');
  assert.equal(f.owner.getTextureTarget(source.texture), source);
  f.owner.withNativeTexture(input, texture => assert.equal(texture.dimension, '3d'));
  assert.equal(source.texture.renderTarget, null);
});

test('authored volume mip counts, byte budgets, slice bounds and native sampler W/LOD follow XYZ storage', t => {
  const f = fixture(), source = new THREE.RenderTarget3D(1, 1, 16, {depthBuffer: false}); t.after(() => f.owner.dispose());
  source.texture.mipmaps = Array.from({length: 5}, () => ({}));
  source.texture.wrapR = THREE.RepeatWrapping;
  const entry = f.owner.capture(source), input = f.owner.captureNativeTexture(source.texture);
  assert.equal(entry.shape.mipLevelCount, 5); assert.equal(entry.shape.bytes, 4 * (16 + 8 + 4 + 2 + 1));
  assert.equal(input.sampler.addressModeW, 'repeat'); assert.equal(input.sampler.lodMaxClamp, 4);
  assert.deepEqual(inspectThreeRenderTargetMip(entry.shape, 3, 1), {width: 1, height: 1});
  for (const [mipLevel, face] of [[0, 16], [1, 8], [2, 4], [3, 2], [4, 1]])
    assert.throws(() => inspectThreeRenderTargetMip(entry.shape, mipLevel, face), {code: 'THREE_TARGET_FACE'});
  f.owner.render(entry, frame => {
    assert.equal(frame.depthSlice, 1); assert.equal(frame.colorView.descriptor.baseMipLevel, 3);
    assert.equal(frame.depthView, undefined);
  }, {mipLevel: 3, face: 1});
  assert.equal(f.owner.allocatedBytes, entry.shape.bytes);
});

test('source depth resize replaces the volume generation and normalizes the same public depth texture', t => {
  const f = fixture(), source = target({useArrayDepthTexture: true}); t.after(() => f.owner.dispose());
  const depth = source.depthTexture = new THREE.DepthTexture(8, 4);
  const before = f.owner.capture(source), input = f.owner.captureNativeTexture(source.texture);
  f.owner.render(before, () => {}, {face: 4}); const old = f.calls.textures.slice();
  source.setSize(4, 2, 1);
  assert.equal(depth.image.depth, 5, 'source depth dimensions are initialized lazily');
  assert.throws(() => f.owner.validateNativeTextures([input]), {code: 'THREE_TARGET_STALE'});
  const after = f.owner.capture(source), sampledDepth = f.owner.captureNativeTexture(depth);
  assert.deepEqual([after.shape.volume, after.shape.layers, after.shape.depthLayers, sampledDepth.viewDimension], [true, 1, 1, '2d']);
  f.owner.render(after, frame => assert.equal(frame.depthSlice, 0));
  assert.equal(source.depthTexture, depth); assert.equal(depth.isArrayTexture, false);
  assert.deepEqual(depth.image, {width: 4, height: 2, depth: 1});
  assert.ok(old.every(texture => texture.destroyed === 1));
  assert.equal(f.owner.allocatedBytes, 4 * 2 * 8);
});

test('volume source and device admission fail before storage allocation', t => {
  const f = fixture(); t.after(() => f.owner.dispose());
  for (const source of [target({samples: 4}), target({generateMipmaps: true}), target({multiview: true})])
    assert.throws(() => inspectThreeRenderTarget(source, THREE), {code: 'THREE_TARGET_UNSUPPORTED'});
  const mismatch = target(); mismatch.texture.image.depth = 4;
  assert.throws(() => inspectThreeRenderTarget(mismatch, THREE), {code: 'THREE_TARGET_SIZE'});
  const large = new THREE.RenderTarget3D(4, 2, 257), input = f.owner.captureNativeTexture(large.texture);
  assert.throws(() => f.owner.validateNativeTextures([input]), {code: 'THREE_TARGET_DEVICE'});
  assert.equal(f.calls.textures.length, 0);
});

test('volume transfer preflight includes the independent depth attachment dimension limit', t => {
  const f = fixture(); t.after(() => f.owner.dispose());
  f.device.limits.maxTextureDimension2D = 4;
  const input = f.owner.captureNativeTexture(target().texture);
  assert.throws(() => f.owner.validateNativeTextures([input]), {code: 'THREE_TARGET_DEVICE'});
  assert.equal(f.calls.textures.length, 0);
  const color = f.owner.captureNativeTexture(target({depthBuffer: false}).texture);
  f.owner.validateNativeTextures([color]);
  f.owner.withNativeTexture(color, texture => assert.equal(texture.size.width, 8));
  assert.equal(f.calls.textures.length, 1);
});

test('color-only source volumes retain their 3D extent without allocating array depth', t => {
  const f = fixture(); t.after(() => f.owner.dispose());
  f.device.limits.maxTextureArrayLayers = 2;
  const source = target({depthBuffer: false, useArrayDepthTexture: true});
  const input = f.owner.captureNativeTexture(source.texture);
  f.owner.validateNativeTextures([input]);
  f.owner.withNativeTexture(input, texture => assert.equal(texture.size.depthOrArrayLayers, 5));
  assert.equal(f.calls.textures.length, 1);
  const attached = f.owner.captureNativeTexture(target({useArrayDepthTexture: true}).texture);
  assert.throws(() => f.owner.validateNativeTextures([attached]), {code: 'THREE_TARGET_DEVICE'});
  assert.equal(f.calls.textures.length, 1);
});
