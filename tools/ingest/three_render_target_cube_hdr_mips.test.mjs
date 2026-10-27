/** Exact source objects at recorded native boundaries. Byte-copy fixtures check
 * subresources and precision; they do not execute WGSL or claim GPU parity. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {createThreeRenderTargets, inspectThreeRenderTarget} from './three_render_targets.mjs';
import {createThreeComputeTargets} from './three_compute_targets.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {targetDevice} from './fixtures/target_residency_fixture.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {mrtTransferDevice, bytes, layouts} from './fixtures/mrt_transfer_fixture.mjs';

const formats = [
  ['rgba8unorm', T.UnsignedByteType, T.RGBAFormat, T.NoColorSpace, 4],
  ['rgba8unorm-srgb', T.UnsignedByteType, T.RGBAFormat, T.SRGBColorSpace, 4],
  ['rgba16float', T.HalfFloatType, T.RGBAFormat, T.NoColorSpace, 8],
  ['r32float', T.FloatType, T.RedFormat, T.NoColorSpace, 4],
  ['rg32float', T.FloatType, T.RGFormat, T.NoColorSpace, 8],
  ['rgba32float', T.FloatType, T.RGBAFormat, T.NoColorSpace, 16],
];
function cube(size = 4, levels = 3, options = {}) {
  const target = new T.CubeRenderTarget(size, {type: T.HalfFloatType,
    generateMipmaps: false, minFilter: T.NearestMipmapNearestFilter, magFilter: T.NearestFilter, ...options});
  // Pinned Textures.updateTexture adds the base level to uncompressed cubes.
  target.texture.mipmaps = Array.from({length: levels - 1}, () => ({}));
  return target;
}
function fixture(t, options = {}) {
  const f = targetDevice(); f.device.limits.maxTextureArrayLayers = 256;
  const owner = createThreeRenderTargets(T, {getDevice: () => f.device, ...options});
  const inputs = createThreeComputeTargets({targets: owner, device: f.device});
  t.after(() => { inputs.dispose(); owner.dispose(); });
  return {...f, owner, inputs};
}

for (const [format, type, sourceFormat, colorSpace, bpp] of formats)
  test(format + ': source cube authored levels preserve format, budget, sampler and native view', async t => {
    const f = fixture(t), target = cube(4, 3, {type, format: sourceFormat, colorSpace});
    const faces = target.texture.image, descriptors = target.texture.mipmaps;
    const entry = f.owner.capture(target), shape = entry.shape;
    assert.equal(shape.mipLevelCount, 3); assert.equal(shape.format, format);
    assert.equal(shape.colorBytes, (16 + 4 + 1) * 6 * bpp);
    assert.equal(shape.bytes, (16 + 4 + 1) * 6 * bpp + 16 * 6 * 4);
    const source = f.inputs.captureBinding({texture: target.texture, isSampledTexture: true});
    const sampler = f.inputs.captureBinding({texture: target.texture, isSampler: true});
    assert.equal(f.calls.textures.length, 0);
    assert.deepEqual(f.inputs.layout(source), {texture: {sampleType: type === T.FloatType ? 'unfilterable-float' : 'float',
      viewDimension: 'cube', multisampled: false}});
    f.owner.render(entry, frame => {
      assert.equal(frame.colorView.texture.format, format);
      assert.equal(frame.colorView.descriptor.baseMipLevel, 2);
      assert.equal(frame.colorView.descriptor.baseArrayLayer, 5);
    }, {mipLevel: 2, face: 5});
    f.inputs.withDispatch([source, sampler], () => {
      const view = f.inputs.resolve(source), state = f.inputs.resolve(sampler);
      assert.deepEqual(view.descriptor, {dimension: 'cube', baseMipLevel: 0, mipLevelCount: 3,
        baseArrayLayer: 0, arrayLayerCount: 6});
      assert.equal(view.texture, f.calls.textures[0]); assert.equal(view.texture.format, format);
      assert.equal(state.lodMaxClamp, 2); assert.equal(state.minFilter, 'nearest');
    });
    assert.equal(target.texture.renderTarget, null); assert.equal(target.texture.image, faces);
    assert.equal(target.texture.mipmaps, descriptors); assert.equal(entry.sample, null);
    assert.equal(f.owner.allocatedBytes, shape.bytes + 6 * 4);
    f.inputs.release(source); f.inputs.release(sampler); await f.inputs.whenIdle();
  });

test('authored cube faces and mip levels have independent color views and matching shared private depth', async t => {
  const f = fixture(t), target = createGpuRenderTarget(f.device, {width: 8, height: 8,
    format: 'rgba16float', cube: true, mipLevelCount: 4, generateMipmaps: false});
  t.after(() => target.dispose());
  const frames = new Set(), depths = new Map();
  assert.equal(target.allocatedBytes, (64 + 16 + 4 + 1) * 6 * 8 + 64 * 6 * 4);
  for (let level = 0; level < 4; level++) for (let face = 0; face < 6; face++) {
    const extent = 8 >> level;
    assert.equal(target.frameAllocationBytes(level, face), level > 0 && face === 0 ? extent * extent * 6 * 4 : 0);
    target.withFrame(frame => {
      frames.add(frame);
      assert.equal(frame.colorView.descriptor.baseMipLevel, level);
      assert.equal(frame.colorView.descriptor.mipLevelCount, 1);
      for (const view of [frame.colorView, frame.depthView]) {
        assert.equal(view.descriptor.dimension, '2d');
        assert.equal(view.descriptor.baseArrayLayer, face); assert.equal(view.descriptor.arrayLayerCount, 1);
      }
      const depth = frame.depthView.texture;
      assert.deepEqual([depth.width, depth.height, depth.size.depthOrArrayLayers], [extent, extent, 6]);
      if (depths.has(level)) assert.equal(depth, depths.get(level)); else depths.set(level, depth);
    }, {mipLevel: level, face});
  }
  assert.equal(frames.size, 24); assert.equal(new Set(depths.values()).size, 4);
  assert.equal(f.calls.textures.length, 5);
  assert.equal(target.allocatedBytes, (64 + 16 + 4 + 1) * 6 * (8 + 4));
  target.withFrame(frame => assert.ok(frames.has(frame)), {mipLevel: 2, face: 3});
  target.withTexture(texture => assert.equal(texture.mipLevelCount, 4));
  assert.equal(target.diagnostics.mipGenerations, 0); assert.equal(f.calls.submissions.length, 0);
  await target.whenIdle(); target.dispose();
  assert.ok(f.calls.textures.every(texture => texture.destroyed === 1));
});

test('source aggregate budget reserves all six private mip depth faces before allocation and can retry', async t => {
  const source = cube(), cost = inspectThreeRenderTarget(source, T).bytes;
  const f = fixture(t, {maxBytes: cost + 2 * 2 * 6 * 4}), entry = f.owner.capture(source);
  const other = new T.RenderTarget(1, 1, {depthBuffer: false});
  f.owner.ensure(entry); f.owner.ensure(f.owner.capture(other));
  const count = f.calls.textures.length;
  assert.throws(() => f.owner.render(entry, () => assert.fail('budget must precede callback'), {mipLevel: 1, face: 4}),
    {code: 'THREE_TARGET_BUDGET'});
  assert.equal(f.calls.textures.length, count); assert.equal(entry.alive, true);
  other.dispose();
  f.owner.render(entry, frame => assert.equal(frame.depthView.texture.size.depthOrArrayLayers, 6), {mipLevel: 1, face: 4});
  assert.equal(f.owner.allocatedBytes, cost + 2 * 2 * 6 * 4);
  await f.owner.whenIdle();
});

test('depth-free cube mips allocate only color and unsupported counts/depth combinations fail before extra allocation', t => {
  const f = fixture(t), target = cube(4, 3, {depthBuffer: false}), entry = f.owner.capture(target);
  f.owner.render(entry, frame => assert.equal(frame.depthView, undefined), {mipLevel: 2, face: 5});
  assert.equal(f.calls.textures.length, 1); assert.equal(f.owner.allocatedBytes, (16 + 4 + 1) * 6 * 8);
  for (const level of [-1, .5, 3])
    assert.throws(() => f.owner.render(entry, () => assert.fail(), {mipLevel: level, face: 0}), {code: 'THREE_TARGET_MIP'});
  assert.throws(() => f.owner.capture(cube(4, 4)), {code: 'THREE_TARGET_SIZE'});
  assert.throws(() => f.owner.capture(cube(4, 3, {generateMipmaps: true})), {code: 'THREE_TARGET_UNSUPPORTED'});
  const sampledDepth = cube(); sampledDepth.depthTexture = new T.CubeDepthTexture(4);
  const depthEntry = f.owner.capture(sampledDepth), count = f.calls.textures.length;
  assert.throws(() => f.owner.render(depthEntry, () => assert.fail(), {mipLevel: 1, face: 0}), {code: 'THREE_TARGET_MIP'});
  assert.equal(f.calls.textures.length, count);
});

test('float32 cube filtering remains feature-gated without lowering attachment precision', t => {
  const f = fixture(t), target = cube(4, 3, {type: T.FloatType, minFilter: T.LinearMipmapLinearFilter});
  f.owner.capture(target);
  assert.throws(() => f.inputs.captureBinding({texture: target.texture, isSampledTexture: true}), {code: 'THREE_TARGET_SAMPLER'});
  assert.equal(f.calls.textures.length, 0); assert.equal(target.texture.type, T.FloatType);
  f.device.features.add('float32-filterable');
  const source = f.inputs.captureBinding({texture: target.texture, isSampledTexture: true});
  assert.equal(f.inputs.layout(source).texture.sampleType, 'float'); f.inputs.release(source);
});

test('source cube mip-count changes invalidate retained views before new storage is allocated', t => {
  const f = fixture(t), target = cube(), entry = f.owner.capture(target);
  const held = f.inputs.captureBinding({texture: target.texture, isSampledTexture: true});
  f.owner.render(entry, () => {}, {mipLevel: 2, face: 1});
  const old = f.calls.textures.slice(); target.texture.mipmaps.pop();
  assert.throws(() => f.inputs.withDispatch([held], () => assert.fail()), {code: 'THREE_TARGET_STALE'});
  const replacement = f.owner.capture(target); assert.equal(replacement.shape.mipLevelCount, 2);
  assert.ok(old.every(texture => texture.destroyed === 1)); assert.equal(f.owner.allocatedBytes, 0);
  assert.equal(f.calls.textures.length, old.length); f.inputs.release(held);
});

for (const [format, type, sourceFormat, colorSpace] of formats)
  test(format + ': public cube mip readback preserves selected face, level and typed raw bytes', async t => {
    const f = mrtTransferDevice(), Base = createWebGPURendererClass(T);
    class Renderer extends Base {
      async _init() { this._device = f.device; this._initialized = true; return this; }
    }
    const renderer = new Renderer({canvas: {width: 8, height: 8, style: {}}});
    t.after(() => renderer.dispose()); await renderer.init();
    const target = cube(4, 3, {type, format: sourceFormat, colorSpace, depthBuffer: false});
    const entry = renderer._targets.capture(target), [Type, components] = layouts.get(format);
    renderer.setRenderTarget(target, 4, 1);
    for (let level = 0; level < 3; level++) for (let face = 0; face < 6; face++) {
      const extent = 4 >> level, expected = new Type(extent * extent * components);
      for (let i = 0; i < expected.length; i++) expected[i] = Type === Float32Array
        ? (i % 2 ? -.75 : 2.5) + face * 4 + level / 4
        : Type === Uint16Array ? [0xc000, 0x4100, 0x3c00, 0xbc00][(i + face + level) % 4] : (i + face * 31 + level * 17) % 256;
      // Populate the independent device byte fixture, without pretending that
      // its recorded render passes execute shaders or write GPU pixels.
      renderer._targets.render(entry, frame => frame.colorView.texture.planes[level][face].set(bytes(expected)), {mipLevel: level, face});
      const actual = await renderer.readRenderTargetMipPixelsAsync(target, level, {face});
      assert.ok(actual instanceof Type); assert.deepEqual(bytes(actual), bytes(expected));
      assert.equal(f.reads.at(-1).source.mipLevel, level); assert.equal(f.reads.at(-1).source.origin.z, face);
    }
    assert.equal(renderer.getActiveCubeFace(), 4); assert.equal(renderer.getActiveMipmapLevel(), 1);
    assert.equal(f.calls.textures.length, 1);
  });

test('source CubeCamera can author HDR cube LODs and feed the same full chain into native graphics', async t => {
  const device = geometryDevice(); device.features = new Set(); device.textures = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => {
    const texture = {...descriptor, width: descriptor.size.width ?? descriptor.size[0],
      height: descriptor.size.height ?? descriptor.size[1], depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? 1,
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createSampler = descriptor => descriptor;
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  let configuration;
  const canvas = {width: 16, height: 16, style: {}, getContext: () => ({configure(value) { configuration = value; },
    unconfigure() {}, getCurrentTexture() { return device.createTexture({size: [16, 16], format: configuration.format, usage: 16}); }})};
  const Renderer = createWebGPURendererClass(T), renderer = new Renderer({canvas, device});
  t.after(() => renderer.dispose()); await renderer.init();
  const target = cube(8, 4), camera = new T.CubeCamera(.1, 10, target), scene = new T.Scene();
  const material = new T.MeshBasicNodeMaterial({side: T.BackSide}); material.colorNode = T.TSL.vec3(2.5, -.75, 4);
  scene.add(new T.Mesh(new T.BoxGeometry(4, 4, 4), material));
  for (const level of [1, 2, 0]) { camera.activeMipmapLevel = level; camera.update(renderer, scene); }
  const consumer = new T.Scene(), consumerMaterial = new T.MeshBasicNodeMaterial();
  consumerMaterial.colorNode = T.TSL.cubeTexture(target.texture, T.TSL.vec3(1, 0, 0)).level(1);
  consumer.add(new T.Mesh(new T.PlaneGeometry(), consumerMaterial));
  const viewCamera = new T.PerspectiveCamera(50, 1, .1, 10); viewCamera.position.z = 3;
  renderer.setRenderTarget(new T.RenderTarget(4, 4, {depthBuffer: false})); renderer.render(consumer, viewCamera);
  await renderer.waitForGPU();
  const native = device.textures.find(texture => texture.depthOrArrayLayers === 6 && texture.format === 'rgba16float');
  assert.ok(native); assert.equal(native.mipLevelCount, 4);
  const writes = device.passes.filter(pass => pass.desc.colorAttachments[0].view.texture === native);
  assert.equal(writes.length, 18);
  assert.deepEqual(writes.map(pass => pass.desc.colorAttachments[0].view.baseMipLevel),
    [1, 2, 0].flatMap(level => Array(6).fill(level)));
  assert.deepEqual(writes.map(pass => pass.desc.colorAttachments[0].view.baseArrayLayer), Array(3).fill([0, 1, 2, 3, 4, 5]).flat());
  const reads = device.snapshots.flat().flatMap(draw => [...draw.groups.values()].flatMap(({group}) => group.entries.map(binding => binding.resource)))
    .filter(view => view?.texture === native);
  assert.equal(reads.length, 1); assert.equal(reads[0].dimension, 'cube'); assert.equal(reads[0].mipLevelCount, 4);
  assert.equal(target.texture.renderTarget, null); assert.equal(target.texture.generateMipmaps, false);
});
