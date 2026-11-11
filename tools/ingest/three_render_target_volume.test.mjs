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
  f.device.createBindGroupLayout = descriptor => descriptor;
  f.device.createPipelineLayout = descriptor => descriptor;
  const draws = [], createEncoder = f.device.createCommandEncoder;
  f.device.createCommandEncoder = () => {
    const encoder = createEncoder(), begin = encoder.beginRenderPass;
    encoder.copyTextureToTexture = (source, destination, size) => {
      (f.calls.copies ??= []).push({source, destination, size});
    };
    encoder.beginRenderPass = descriptor => {
      const pass = begin(descriptor);
      return {...pass, draw(...args) { draws.push({descriptor, args}); pass.draw(...args); }};
    };
    return encoder;
  };
  return {...f, draws, owner: createThreeRenderTargets(THREE, {getDevice: () => f.device, ...options})};
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

for (const type of [THREE.UnsignedByteType, THREE.FloatType])
test(`WebGL volume sampling preserves all slices in a full-precision 3D GPU replica (type=${type})`, async t => {
  const f = fixture(), source = new THREE.WebGL3DRenderTarget(8, 4, 5, {depthBuffer: false, type,
    minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter});
  t.after(() => f.owner.dispose()); t.after(() => source.dispose()); source.texture.wrapR = THREE.RepeatWrapping;
  const entry = f.owner.capture(source), bindings = f.owner.bindingsFor();
  f.owner.render(entry, () => {}, {face: 4}); const binding = bindings.get(source.texture);
  assert.equal(binding.viewDimension, '3d'); assert.equal(binding.view.descriptor.dimension, '3d');
  assert.equal(binding.view.texture.dimension, '3d');
  assert.deepEqual(binding.view.texture.size, {width: 8, height: 4, depthOrArrayLayers: 5});
  assert.equal(binding.view.texture.format, type === THREE.FloatType ? 'rgba32float' : 'rgba8unorm');
  assert.equal(binding.sampler.addressModeW, 'repeat'); assert.equal(binding.sampler.lodMaxClamp, 0);
  assert.equal(f.owner.allocatedBytes, 2 * entry.shape.colorBytes);
  const passes = f.calls.submissions.at(-1).passes;
  assert.deepEqual(passes.map(pass => pass.descriptor.colorAttachments[0].depthSlice), [0, 1, 2, 3, 4]);
  assert.deepEqual(f.draws.map(draw => draw.args), [0, 1, 2, 3, 4].map(slice => [3, 1, 0, slice]));
  for (const pass of passes) {
    const output = pass.descriptor.colorAttachments[0].view, input = pass.group.entries[0].resource;
    assert.notEqual(output.texture, input.texture); assert.deepEqual(output.descriptor, input.descriptor);
    assert.deepEqual(output.descriptor, {baseMipLevel: 0, mipLevelCount: 1, dimension: '3d'});
  }
  assert.match(f.calls.modules[0].code, /texture_3d<f32>/);
  assert.match(f.calls.modules[0].code, /i32\(size.y\) - 1 - i32\(v.position.y\)/);
  if (type === THREE.FloatType) {
    assert.equal(binding.sampleType, 'unfilterable-float');
    assert.equal(f.calls.pipelines[0].layout.bindGroupLayouts[0].entries[0].texture.viewDimension, '3d');
  }
  assert.equal(bindings.get(source.texture).view, binding.view); assert.equal(entry.copies, 1);
  f.owner.render(entry, () => {}, {face: 1});
  assert.equal(bindings.get(source.texture).view, binding.view); assert.equal(entry.copies, 2);
  assert.throws(() => f.owner.bindingsFor(source).get(source.texture), {code: 'THREE_TARGET_FEEDBACK'});
  assert.equal(source.texture.renderTarget, null); await f.owner.whenIdle();
});

test('WebGL volume setup sampler and borrowed generation survive source edits until recreation', async t => {
  const f = fixture(), source = new THREE.WebGL3DRenderTarget(8, 4, 5, {depthBuffer: false,
    minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter});
  t.after(() => f.owner.dispose()); const texture = source.texture; texture.wrapR = THREE.RepeatWrapping;
  f.owner.capture(source); const bindings = f.owner.bindingsFor(), first = bindings.get(texture), release = first.retain();
  const old = f.calls.textures.slice(), oldBytes = f.owner.allocatedBytes;
  texture.minFilter = texture.magFilter = THREE.NearestFilter; texture.wrapR = THREE.ClampToEdgeWrapping; texture.needsUpdate = true;
  assert.equal(bindings.get(texture).sampler, first.sampler);
  source.setSize(4, 2, 3); assert.equal(source.texture, texture);
  assert.throws(first.validate, {code: 'THREE_TARGET_STALE'}); assert.ok(old.every(value => value.destroyed === 0));
  f.owner.capture(source); const next = bindings.get(texture);
  assert.notEqual(next.view, first.view); assert.equal(next.sampler.addressModeW, 'clamp-to-edge');
  assert.equal(next.sampler.minFilter, 'nearest'); assert.equal(next.sampler.magFilter, 'nearest');
  assert.deepEqual(next.view.texture.size, {width: 4, height: 2, depthOrArrayLayers: 3});
  assert.equal(f.owner.allocatedBytes, oldBytes + 2 * 4 * 2 * 3 * 4);
  release(); await Promise.resolve(); await Promise.resolve(); assert.ok(old.every(value => value.destroyed === 1));
  source.dispose(); await f.owner.whenIdle();
  assert.equal(f.owner.allocatedBytes, 0); assert.ok(f.calls.textures.every(value => value.destroyed === 1));
});

test('WebGL volume generation follows captured render flags and orients each shrinking mip slice', async t => {
  const f = fixture(), source = new THREE.WebGL3DRenderTarget(8, 4, 5, {depthBuffer: false,
    generateMipmaps: true, minFilter: THREE.NearestMipmapNearestFilter, magFilter: THREE.NearestFilter});
  t.after(() => f.owner.dispose()); t.after(() => source.dispose());
  const entry = f.owner.capture(source, {initialize: true}), bindings = f.owner.bindingsFor();
  f.owner.render(entry, () => {}, {face: 4, generateColorMipmaps: [0]});
  assert.equal(entry.storage.diagnostics.mipGenerations, 1);
  const binding = bindings.get(source.texture), native = entry.storage;
  assert.equal(binding.view.texture.mipLevelCount, 4); assert.equal(binding.sampler.lodMaxClamp, 3);
  assert.deepEqual(f.calls.submissions.at(-1).passes.map(pass => {
    const attachment = pass.descriptor.colorAttachments[0];
    return [attachment.view.descriptor.baseMipLevel, attachment.depthSlice];
  }), [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1], [2, 0], [3, 0]]);
  assert.equal(f.owner.allocatedBytes, 2 * (160 + 16 + 2 + 1) * 4);
  source.texture.generateMipmaps = false;
  assert.equal(f.owner.capture(source), entry, 'flag changes retain rendered content and initialized sampler');
  f.owner.render(entry, () => {}, {face: 1, generateColorMipmaps: []});
  bindings.get(source.texture);
  assert.equal(entry.storage, native); assert.equal(native.diagnostics.mipGenerations, 1, 'sampling must not regenerate a stale source pyramid');
  source.texture.generateMipmaps = true;
  f.owner.render(entry, () => {}, {face: 2, generateColorMipmaps: [0]});
  assert.equal(native.diagnostics.mipGenerations, 2);
  assert.equal(bindings.get(source.texture).view, binding.view); await f.owner.whenIdle();
});

test('enabling WebGL volume mipmaps grows XYZ storage after the draw without losing base content', async t => {
  const f = fixture(), source = new THREE.WebGL3DRenderTarget(7, 5, 3, {depthBuffer: false});
  t.after(() => f.owner.dispose()); t.after(() => source.dispose());
  const entry = f.owner.capture(source, {initialize: true});
  f.owner.render(entry, () => {}, {face: 2, generateColorMipmaps: []});
  const binding = f.owner.bindingsFor().get(source.texture), previous = binding.view.texture;
  source.texture.generateMipmaps = true;
  assert.equal(f.owner.capture(source), entry);
  let base;
  f.owner.render(entry, (_, color) => { base = color; }, {face: 1, generateColorMipmaps: [0]});
  assert.equal(entry.shape.mipLevelCount, 3); assert.equal(entry.shape.colorBytes, (105 + 6 + 1) * 4);
  assert.equal(f.calls.copies.length, 1);
  assert.equal(f.calls.copies[0].source.texture, base);
  assert.deepEqual(f.calls.copies[0].size, {width: 7, height: 5, depthOrArrayLayers: 3});
  const grown = f.calls.copies[0].destination.texture;
  assert.equal(grown.dimension, '3d'); assert.equal(grown.mipLevelCount, 3);
  const next = f.owner.bindingsFor().get(source.texture);
  assert.notEqual(next.view.texture, previous);
  assert.equal(next.sampler.minFilter, binding.sampler.minFilter, 'the source setup sampler survives storage growth');
  assert.equal(next.view.texture.mipLevelCount, 3);
  f.owner.render(entry, frame => { assert.equal(frame.colorView.texture, grown); assert.equal(frame.depthSlice, 2); },
    {face: 2, generateColorMipmaps: []});
  await f.owner.whenIdle(); assert.equal(f.owner.allocatedBytes, 2 * 112 * 4);
});

test('WebGL volume sampling retains source mip, source-kind, precision and replica-budget boundaries', t => {
  for (const [Type, options, mips, code] of [
    [THREE.RenderTarget3D, {}, 0, 'THREE_TARGET_UNSUPPORTED'],
    [THREE.WebGL3DRenderTarget, {}, 4, 'THREE_TARGET_UNSUPPORTED'],
    [THREE.WebGL3DRenderTarget, {minFilter: THREE.NearestMipmapNearestFilter}, 0, 'THREE_TARGET_UNSUPPORTED'],
    [THREE.WebGL3DRenderTarget, {type: THREE.FloatType, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter}, 0, 'THREE_TARGET_SAMPLER'],
  ]) {
    const f = fixture(), source = new Type(8, 4, 5, {depthBuffer: false, ...options});
    t.after(() => f.owner.dispose()); t.after(() => source.dispose());
    source.texture.mipmaps = Array.from({length: mips}, () => ({})); const entry = f.owner.capture(source);
    assert.throws(() => f.owner.bindingsFor().get(source.texture), {code});
    if (mips) assert.throws(() => inspectThreeRenderTargetMip(entry.shape, 1, 0, {webgl: true}), {code: 'THREE_TARGET_MIP'});
    assert.equal(f.calls.textures.length, 0); assert.equal(f.owner.allocatedBytes, 0);
  }
  const f = fixture({maxBytes: 1279}), source = new THREE.WebGL3DRenderTarget(8, 4, 5, {depthBuffer: false});
  t.after(() => f.owner.dispose()); t.after(() => source.dispose()); f.owner.capture(source);
  assert.throws(() => f.owner.bindingsFor().get(source.texture), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(f.owner.allocatedBytes, 640); assert.equal(f.calls.textures.length, 1); assert.equal(f.calls.submissions.length, 0);
});

for (const Type of [THREE.WebGLArrayRenderTarget, THREE.WebGL3DRenderTarget])
test(`${Type.name} base-only sampling preserves source-disabled anisotropy until recreation`, t => {
  const f = fixture(), source = new Type(8, 4, 3, {depthBuffer: false,
    minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, anisotropy: 8});
  t.after(() => f.owner.dispose()); t.after(() => source.dispose()); f.owner.capture(source);
  const bindings = f.owner.bindingsFor(), first = bindings.get(source.texture);
  assert.equal(first.sampler.minFilter, 'linear'); assert.equal(first.sampler.magFilter, 'linear');
  assert.equal(first.sampler.lodMaxClamp, 0); assert.equal(first.sampler.maxAnisotropy, 1);
  source.texture.anisotropy = 16; source.texture.needsUpdate = true;
  assert.equal(bindings.get(source.texture).sampler, first.sampler);
  source.dispose(); source.setSize(1, 1, 1);
  source.texture.minFilter = THREE.LinearMipmapLinearFilter; source.texture.anisotropy = 8;
  f.owner.capture(source); const next = bindings.get(source.texture);
  assert.equal(next.sampler.maxAnisotropy, 8, 'a complete mip-linear image may enable anisotropy at setup');
  assert.notEqual(next.sampler, first.sampler);
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
  const native = fixture({flipY: false}), legacy = new THREE.WebGL3DRenderTarget(8, 4, 5, {generateMipmaps: true});
  t.after(() => { native.owner.dispose(); legacy.dispose(); });
  assert.throws(() => native.owner.capture(legacy), {code: 'THREE_TARGET_UNSUPPORTED'});
  assert.equal(native.calls.textures.length, 0);
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
