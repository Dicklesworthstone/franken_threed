/** Actual target owners with descriptor/event doubles; no source-class or GPU-pixel claim. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {createThreeRenderTargets, inspectThreeRenderTarget, inspectThreeRenderTargetMip} from './three_render_targets.mjs';
import {createThreeComputeTargets} from './three_compute_targets.mjs';
import {targetThree} from './fixtures/target_residency_fixture.mjs';
import {gpuTargetMipmapFixture, deferred} from './fixtures/animation/gpu_target_mipmap_fixture.mjs';

class Data3DTexture extends targetThree.Texture {
  isData3DTexture = true;
  constructor(width, height, depth) {
    super(); this.image = {width, height, depth}; this.wrapR = targetThree.ClampToEdgeWrapping;
  }
}
const T = {...targetThree, Data3DTexture, NearestMipmapNearestFilter: 1004};
function fixture(options = {}) {
  const f = gpuTargetMipmapFixture();
  Object.assign(f.device.limits, {maxTextureDimension3D: 256, maxTextureArrayLayers: 256,
    maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  f.device.features = new Set();
  return {...f, owner: createThreeRenderTargets(T, {getDevice: () => f.device, flipY: false, ...options})};
}
function color(target, index) {
  const texture = new T.Data3DTexture(target.width, target.height, target.depth);
  Object.assign(texture, {isRenderTargetTexture: true, renderTarget: target,
    type: index === 0 ? T.UnsignedByteType : T.HalfFloatType,
    minFilter: T.NearestMipmapNearestFilter, magFilter: T.NearestFilter,
    generateMipmaps: false, mipmaps: Array.from({length: index === 0 ? 4 : 3}, () => ({}))});
  return texture;
}
function target(options = {}) {
  const target = new T.RenderTarget(8, 4, {depth: 5, depthBuffer: false, ...options});
  target.isRenderTarget3D = true;
  target.textures = [color(target, 0), color(target, 1)];
  return target;
}
function resize(target, width, height) {
  target.width = width; target.height = height;
  for (const texture of target.textures) Object.assign(texture.image, {width, height});
  target.dispose();
}
const colorBytes = (160 + 16 + 2 + 1) * 4 + (160 + 16 + 2) * 8;
const destroyed = f => f.textures.forEach(texture => assert.equal(texture.destroyed, 1));

test('native volume MRT accounts for separate XYZ mip chains and samples each as a genuine 3D view', async () => {
  const f = fixture(), source = target(), shape = inspectThreeRenderTarget(source, T);
  assert.equal(shape.bytes, colorBytes);
  assert.equal(shape.viewDimension, '3d');
  assert.deepEqual(shape.colorMipLevelCounts, [4, 3]);
  assert.deepEqual(shape.colorFormats, ['rgba8unorm', 'rgba16float']);
  const sampled = createThreeComputeTargets({targets: f.owner, device: f.device});
  const bindings = source.textures.map(texture => sampled.captureBinding({texture, isSampledTexture: true}));
  assert.ok(bindings.every(binding => sampled.layout(binding).texture.viewDimension === '3d'));
  sampled.withDispatch(bindings, () => {
    const views = bindings.map(binding => sampled.resolve(binding));
    assert.deepEqual(views.map(view => view.texture.dimension), ['3d', '3d']);
    assert.deepEqual(views.map(view => view.texture.depthOrArrayLayers), [5, 5]);
    assert.deepEqual(views.map(view => view.descriptor), [
      {dimension: '3d', baseMipLevel: 0, mipLevelCount: 4},
      {dimension: '3d', baseMipLevel: 0, mipLevelCount: 3},
    ]);
    assert.notEqual(views[0].texture, views[1].texture);
  });
  assert.equal(f.owner.allocatedBytes, colorBytes);
  assert.equal(f.textures.length, 2, 'sampling borrows the original volumes');
  assert.equal(f.submissions.length, 0, 'authored mip sampling must not launch a reduction');
  for (const binding of bindings) sampled.release(binding);
  await sampled.whenIdle(); sampled.dispose(); await f.owner.whenIdle(); f.owner.dispose(); destroyed(f);
});

test('volume MRT render selectors use mip-local depth and the common attachment mip range', async () => {
  const f = fixture(), entry = f.owner.capture(target());
  for (const [mipLevel, face, code] of [[1, 2, 'THREE_TARGET_FACE'], [2, 1, 'THREE_TARGET_FACE'],
    [3, 0, 'THREE_TARGET_MIP']]) {
    assert.throws(() => f.owner.render(entry, () => assert.fail('unallocated slice or mip'), {mipLevel, face}), {code});
  }
  assert.equal(f.textures.length, 0);
  for (const [mipLevel, face, width, height] of [[0, 4, 8, 4], [1, 1, 4, 2], [2, 0, 2, 1]]) {
    assert.deepEqual(inspectThreeRenderTargetMip(entry.shape, mipLevel, face), {width, height});
    f.owner.render(entry, frame => {
      assert.equal(frame.depthSlice, face);
      assert.deepEqual(frame.colorViews.map(view => view.descriptor), [
        {dimension: '3d', baseMipLevel: mipLevel, mipLevelCount: 1},
        {dimension: '3d', baseMipLevel: mipLevel, mipLevelCount: 1},
      ]);
      assert.equal(frame.depthView, undefined);
    }, {mipLevel, face});
  }
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f);
});

test('volume MRT rejects an over-budget combined chain before allocating either attachment', () => {
  const f = fixture({maxBytes: colorBytes - 1}), source = target(), entry = f.owner.capture(source);
  assert.throws(() => f.owner.ensure(entry), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(f.owner.allocatedBytes, 0); assert.equal(f.textures.length, 0);
  f.owner.dispose();
});

for (const depthArray of [false, true]) test(`volume MRT keeps ${depthArray ? 'array' : 'shared 2D'} private depth separate from shrinking color slices`, async () => {
  const depthLayers = depthArray ? 5 : 1, baseBytes = colorBytes + 8 * 4 * depthLayers * 4;
  const mipBytes = 4 * 2 * depthLayers * 4;
  const source = target({depthBuffer: true, useArrayDepthTexture: depthArray});
  const limited = fixture({maxBytes: baseBytes + mipBytes - 1}), rejected = limited.owner.capture(source);
  assert.throws(() => limited.owner.render(rejected, () => assert.fail('private depth exceeds budget'),
    {mipLevel: 1, face: 1}), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(limited.owner.allocatedBytes, baseBytes);
  assert.equal(limited.textures.length, 3, 'no higher-mip depth was allocated');
  limited.owner.dispose(); destroyed(limited);
  const f = fixture({maxBytes: baseBytes + mipBytes}), entry = f.owner.capture(source);
  let depth;
  for (const face of [0, 1]) f.owner.render(entry, frame => {
    assert.equal(frame.depthSlice, face);
    assert.equal(frame.depthView.texture.dimension, '2d');
    assert.deepEqual(frame.depthView.texture.size, {width: 4, height: 2, depthOrArrayLayers: depthLayers});
    if (depthArray) assert.equal(frame.depthView.descriptor.baseArrayLayer, face);
    if (depth) assert.equal(frame.depthView.texture, depth);
    depth = frame.depthView.texture;
  }, {mipLevel: 1, face});
  assert.equal(f.owner.allocatedBytes, baseBytes + mipBytes);
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f);
});

test('volume MRT typed uploads and copies preserve selected color precision and mip-local slice origins', async () => {
  const f = fixture(), writes = [], copies = [], createEncoder = f.device.createCommandEncoder;
  f.device.queue.writeTexture = (destination, data, layout, size) => writes.push({destination, data, layout, size});
  f.device.createCommandEncoder = () => ({...createEncoder(),
    copyTextureToTexture(source, destination, size) { copies.push({source, destination, size}); }});
  const a = f.owner.capture(target()), b = f.owner.capture(target());
  f.owner.writePixels(a, new Uint16Array(4 * 2 * 4).fill(0x3c00), {colorIndex: 1, destinationMipLevel: 1, face: 1});
  assert.equal(writes.length, 1);
  assert.equal(writes[0].destination.texture.format, 'rgba16float');
  assert.equal(writes[0].destination.texture.dimension, '3d');
  assert.equal(writes[0].destination.mipLevel, 1);
  assert.deepEqual(writes[0].destination.origin, {x: 0, y: 0, z: 1});
  assert.deepEqual(writes[0].size, {width: 4, height: 2, depthOrArrayLayers: 1});
  f.owner.copyColor(a, b, {sourceColorIndex: 1, colorIndex: 1, sourceMipLevel: 1,
    destinationMipLevel: 2, sourceFace: 1, face: 0, width: 2, height: 1});
  assert.equal(copies.length, 1);
  assert.equal(copies[0].source.texture, writes[0].destination.texture);
  assert.equal(copies[0].destination.texture.format, 'rgba16float');
  assert.equal(copies[0].destination.mipLevel, 2);
  assert.deepEqual(copies[0].source.origin, {x: 0, y: 0, z: 1});
  assert.deepEqual(copies[0].destination.origin, {x: 0, y: 0, z: 0});
  assert.throws(() => f.owner.copyColor(a, b, {sourceColorIndex: 1, colorIndex: 1, sourceMipLevel: 1,
    sourceFace: 2, face: 0}), {code: 'GPU_TRANSFER_FACE'});
  assert.equal(copies.length, 1);
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f);
});

test('accepted volume MRT sampling generations survive resize until leases and fences finish', async () => {
  const f = fixture(), source = target(), sampled = createThreeComputeTargets({targets: f.owner, device: f.device});
  const bindings = source.textures.map(texture => sampled.captureBinding({texture, isSampledTexture: true}));
  let oldViews;
  sampled.withDispatch(bindings, () => { oldViews = bindings.map(binding => sampled.resolve(binding)); });
  const original = f.textures.slice(), oldBytes = f.owner.allocatedBytes, fence = deferred();
  f.controls.completion = fence;
  f.owner.withNativePassResize(source, 16, 8, () => resize(source, 16, 8));
  const next = f.owner.capture(source);
  f.owner.render(next, frame => assert.equal(frame.colorViews[0].texture.width, 16), {face: 4});
  sampled.withDispatch(bindings, () => {
    assert.deepEqual(bindings.map(binding => sampled.resolve(binding)), oldViews);
    assert.ok(oldViews.every(view => view.texture.width === 8 && view.texture.depthOrArrayLayers === 5));
  });
  assert.equal(f.owner.allocatedBytes, oldBytes + next.shape.bytes);
  for (const binding of bindings) sampled.release(binding);
  assert.ok(original.every(texture => texture.destroyed === 0));
  fence.resolve(); await sampled.whenIdle(); await f.owner.whenIdle(); await f.owner.whenIdle();
  assert.ok(original.every(texture => texture.destroyed === 1));
  assert.equal(f.owner.allocatedBytes, next.shape.bytes);
  sampled.dispose(); f.owner.dispose(); destroyed(f);
});

test('volume MRT continues to refuse automatic mips, multisampling and history swaps before allocation', () => {
  for (const edit of [source => { source.texture.generateMipmaps = true; },
    source => { source.textures[1].generateMipmaps = true; }, source => { source.samples = 4; }]) {
    const f = fixture(), source = target(); edit(source);
    assert.throws(() => f.owner.capture(source), {code: 'THREE_TARGET_UNSUPPORTED'});
    assert.equal(f.textures.length, 0); f.owner.dispose();
  }
  const f = fixture(), source = target();
  assert.throws(() => f.owner.registerNativePass({renderTarget: source, _textures: {output: source.texture},
    _previousTextures: {output: color(source, 0)}}), {code: 'THREE_TARGET_UNSUPPORTED'});
  assert.equal(f.textures.length, 0); f.owner.dispose();
});
