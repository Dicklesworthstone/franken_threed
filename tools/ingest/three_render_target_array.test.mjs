/** Exact pinned source descriptors + recording GPU ownership, not pixel parity. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeRenderTargets, inspectThreeRenderTarget, inspectThreeRenderTargetMip} from './three_render_targets.mjs';
import {targetDevice} from './fixtures/target_residency_fixture.mjs';

function fixture(options = {}) {
  const f = targetDevice();
  Object.assign(f.device.limits, {maxTextureArrayLayers: 256, maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  return {...f, owner: createThreeRenderTargets(THREE, {getDevice: () => f.device, ...options})};
}
const target = options => new THREE.RenderTarget(8, 4, {depth: 3, ...options});

test('source layered MRT preserves color layers and default shared depth with native array sampling', async t => {
  const f = fixture(), source = target({count: 2}); t.after(() => f.owner.dispose());
  source.textures[1].type = THREE.HalfFloatType; source.textures[1].format = THREE.RGFormat;
  const entry = f.owner.capture(source), {shape} = entry;
  assert.equal(f.calls.textures.length, 0);
  assert.equal(shape.array, true); assert.equal(shape.layers, 3); assert.equal(shape.depthLayers, 1);
  assert.equal(shape.viewDimension, '2d-array'); assert.equal(shape.depthViewDimension, '2d');
  assert.deepEqual(shape.colorFormats, ['rgba8unorm', 'rg16float']); assert.equal(shape.bytes, 8 * 4 * (3 * 8 + 4));
  const frames = [];
  for (let face = 0; face < 3; face++) f.owner.render(entry, frame => {
    frames.push(frame);
    assert.deepEqual(frame.colorViews.map(view => view.descriptor.baseArrayLayer), [face, face]);
  }, {face});
  assert.ok(frames.every(frame => frame.depthView === frames[0].depthView));
  const inputs = source.textures.map(texture => f.owner.captureNativeTexture(texture));
  assert.ok(inputs.every(input => input.viewDimension === '2d-array'));
  f.owner.withNativeTextures(inputs, textures => {
    assert.equal(textures.size, 2);
    for (const texture of source.textures) assert.equal(textures.get(texture).size.depthOrArrayLayers, 3);
  });
  assert.equal(f.calls.textures.length, 3); assert.equal(f.owner.allocatedBytes, shape.bytes);
  assert.throws(() => f.owner.bindingsFor().get(source.texture), {code: 'THREE_TARGET_UNSUPPORTED'});
  await f.owner.whenIdle();
});

for (const authored of [false, true]) test(`source array depth preserves requested depth ownership (authored=${authored})`, async t => {
  const f = fixture(), source = target({useArrayDepthTexture: !authored}); t.after(() => f.owner.dispose());
  source.depthTexture = new THREE.DepthTexture(8, 4, THREE.FloatType);
  if (authored) source.depthTexture.image.depth = 3;
  source.depthTexture.compareFunction = THREE.LessEqualCompare;
  const entry = f.owner.capture(source), input = f.owner.captureNativeTexture(source.depthTexture, {comparison: true});
  assert.equal(entry.shape.depthArray, true); assert.equal(entry.shape.depthLayers, 3);
  assert.equal(entry.shape.depthFormat, 'depth32float'); assert.equal(entry.shape.bytes, 8 * 4 * 3 * 8);
  assert.equal(input.viewDimension, '2d-array'); assert.equal(input.sampleType, 'depth');
  assert.equal(input.sampler.compare, 'less-equal');
  let depth;
  for (let face = 0; face < 3; face++) f.owner.render(entry, frame => {
    depth ??= frame.depthView.texture; assert.equal(frame.depthView.texture, depth);
    assert.equal(frame.depthView.descriptor.baseArrayLayer, face);
    assert.equal(frame.depthView.descriptor.arrayLayerCount, 1);
  }, {face});
  f.owner.withNativeTexture(input, native => assert.equal(native, depth));
  assert.equal(f.calls.textures.length, 2); assert.equal(f.owner.allocatedBytes, entry.shape.bytes);
  await f.owner.whenIdle();
});

test('legacy source array target keeps its null texture owner pointer while native residency identifies it', t => {
  const f = fixture(), source = new THREE.WebGLArrayRenderTarget(8, 4, 3); t.after(() => f.owner.dispose());
  assert.equal(source.texture.renderTarget, null);
  const entry = f.owner.capture(source), input = f.owner.captureNativeTexture(source.texture);
  assert.equal(input.entry, entry); assert.equal(input.viewDimension, '2d-array');
  f.owner.withNativeTexture(input, texture => assert.equal(texture.size.depthOrArrayLayers, 3));
  assert.equal(source.texture.renderTarget, null); assert.equal(source.texture.image.depth, 3);
});

test('source array size/depth changes invalidate earlier generations and reallocate the full new layer count', t => {
  const f = fixture(), source = target({useArrayDepthTexture: true}); t.after(() => f.owner.dispose());
  source.depthTexture = new THREE.DepthTexture(8, 4); source.depthTexture.image.depth = 3;
  const previous = f.owner.capture(source), input = f.owner.captureNativeTexture(source.texture);
  f.owner.render(previous, () => {}, {face: 2}); const old = f.calls.textures.slice();
  source.setSize(4, 2, 5);
  assert.equal(source.depthTexture.image.depth, 3, 'source keeps old depth dimensions until renderer preparation');
  assert.throws(() => f.owner.assertCurrent(previous), {code: 'THREE_TARGET_STALE'});
  assert.throws(() => f.owner.validateNativeTextures([input]), {code: 'THREE_TARGET_STALE'});
  assert.ok(old.every(texture => texture.destroyed === 1));
  const next = f.owner.capture(source);
  assert.deepEqual([next.shape.width, next.shape.height, next.shape.layers, next.shape.depthLayers], [4, 2, 5, 5]);
  f.owner.render(next, frame => assert.equal(frame.colorView.descriptor.baseArrayLayer, 4), {face: 4});
  assert.equal(f.owner.allocatedBytes, 4 * 2 * 5 * 8);
});

test('array source preflight rejects out-of-range layers, invalid dimensions and unsupported target kinds before allocation', t => {
  const f = fixture(); t.after(() => f.owner.dispose());
  const entry = f.owner.capture(target());
  for (const face of [-1, 3, .5]) {
    assert.throws(() => inspectThreeRenderTargetMip(entry.shape, 0, face), {code: 'THREE_TARGET_FACE'});
    assert.throws(() => f.owner.render(entry, () => assert.fail('invalid layer'), {face}), {code: 'THREE_TARGET_FACE'});
  }
  for (const source of [target({multiview: true}), new THREE.RenderTarget3D(8, 4, 3), new THREE.WebGL3DRenderTarget(8, 4, 3)])
    assert.throws(() => inspectThreeRenderTarget(source, THREE), {code: 'THREE_TARGET_UNSUPPORTED'});
  const invalid = target(); invalid.texture.image.depth = 4;
  assert.throws(() => inspectThreeRenderTarget(invalid, THREE), {code: 'THREE_TARGET_SIZE'});
  const sampled = target({samples: 4, useArrayDepthTexture: true}); sampled.depthTexture = new THREE.DepthTexture(8, 4);
  assert.throws(() => inspectThreeRenderTarget(sampled, THREE), {code: 'THREE_TARGET_UNSUPPORTED'});
  assert.equal(f.calls.textures.length, 0);
});

test('array memory and device layer budgets are checked before allocation and remain retryable', t => {
  const f = fixture({maxBytes: 8 * 4 * (3 * 4 + 4) - 1}); t.after(() => f.owner.dispose());
  const source = target(), entry = f.owner.capture(source), input = f.owner.captureNativeTexture(source.texture);
  assert.throws(() => f.owner.validateNativeTextures([input]), {code: 'THREE_TARGET_BUDGET'});
  assert.throws(() => f.owner.ensure(entry), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(entry.alive, true); assert.equal(f.calls.textures.length, 0);
  const limited = fixture(); t.after(() => limited.owner.dispose()); limited.device.limits.maxTextureArrayLayers = 2;
  const captured = limited.owner.captureNativeTexture(target().texture);
  assert.throws(() => limited.owner.validateNativeTextures([captured]), {code: 'THREE_TARGET_DEVICE'});
  assert.equal(limited.calls.textures.length, 0);
});

for (const Type of [THREE.RenderTarget, THREE.WebGLArrayRenderTarget])
test(`${Type.name} with one color layer retains source 2D depth despite array-depth requests or stale depth flags`, t => {
  const f = fixture(); t.after(() => f.owner.dispose());
  const source = Type === THREE.RenderTarget ? new Type(8, 4, {useArrayDepthTexture: true})
    : new Type(8, 4, 1, {useArrayDepthTexture: true});
  source.depthTexture = new THREE.DepthTexture(8, 4); source.depthTexture.image.depth = 3;
  source.depthTexture.isArrayTexture = true;
  const entry = f.owner.capture(source), input = f.owner.captureNativeTexture(source.depthTexture);
  assert.equal(entry.shape.viewDimension, Type === THREE.RenderTarget ? '2d' : '2d-array');
  assert.equal(entry.shape.depthArray, false); assert.equal(entry.shape.depthLayers, 1);
  assert.equal(input.viewDimension, '2d');
  f.owner.render(entry, frame => assert.equal(frame.depthView.texture.size.depthOrArrayLayers ?? 1, 1));
  f.owner.withNativeTexture(input, texture => assert.equal(texture.size.depthOrArrayLayers ?? 1, 1));
  assert.equal(f.owner.allocatedBytes, 8 * 4 * 8);
});

test('source array-to-2D resize replaces depth storage without replacing its public DepthTexture', t => {
  const f = fixture(), source = target({useArrayDepthTexture: true}); t.after(() => f.owner.dispose());
  const depth = source.depthTexture = new THREE.DepthTexture(8, 4);
  depth.image.depth = 3; depth.isArrayTexture = true;
  const previous = f.owner.capture(source), input = f.owner.captureNativeTexture(depth);
  f.owner.render(previous, () => {}, {face: 2}); const old = f.calls.textures.slice();
  source.setSize(8, 4, 1);
  assert.equal(source.depthTexture, depth); assert.equal(depth.image.depth, 3);
  assert.throws(() => f.owner.validateNativeTextures([input]), {code: 'THREE_TARGET_STALE'});
  assert.ok(old.every(texture => texture.destroyed === 1));
  const next = f.owner.capture(source), sampled = f.owner.captureNativeTexture(depth);
  assert.deepEqual([next.shape.array, next.shape.layers, next.shape.depthArray, next.shape.depthLayers], [false, 1, false, 1]);
  assert.equal(sampled.viewDimension, '2d');
  f.owner.render(next, () => {});
  f.owner.withNativeTexture(sampled, texture => assert.equal(texture.size.depthOrArrayLayers ?? 1, 1));
  assert.equal(f.owner.allocatedBytes, 8 * 4 * 8);
});


test('render preparation clears stale authored array depth before a shrink and later regrowth', t => {
  const f = fixture(), source = target(); t.after(() => f.owner.dispose());
  const depth = source.depthTexture = new THREE.DepthTexture(8, 4);
  depth.image.depth = 3;
  const original = f.owner.capture(source);
  f.owner.render(original, () => {}, {face: 2});
  assert.equal(depth.isArrayTexture, true);
  source.setSize(4, 2, 1);
  f.owner.render(f.owner.capture(source), () => {});
  assert.deepEqual(depth.image, {width: 4, height: 2, depth: 1});
  assert.equal(depth.isArrayTexture, false);
  source.setSize(8, 4, 3);
  const grown = f.owner.capture(source);
  assert.equal(grown.shape.layers, 3); assert.equal(grown.shape.depthLayers, 1);
  assert.equal(f.owner.captureNativeTexture(depth).viewDimension, '2d');
  f.owner.render(grown, frame => assert.equal(frame.depthView.texture.size.depthOrArrayLayers, 1), {face: 2});
  assert.equal(depth, source.depthTexture);
  assert.deepEqual(depth.image, {width: 8, height: 4, depth: 1});
});
