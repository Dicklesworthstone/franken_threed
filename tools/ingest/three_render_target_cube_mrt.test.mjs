/** Native cube MRT source contracts and recorded GPU operations. The fixtures
 * do not execute shaders or establish browser-pixel/performance equivalence. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import RenderContext from '../../upstream/three.js/src/renderers/common/RenderContext.js';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {createThreeRenderTargets, inspectThreeRenderTarget} from './three_render_targets.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {gpuTargetMipmapFixture} from './fixtures/animation/gpu_target_mipmap_fixture.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {mrtTransferDevice, bytes, layouts, deferred} from './fixtures/mrt_transfer_fixture.mjs';

function cube(size = 8, levels = [4, 3]) {
  const target = new T.CubeRenderTarget(size, {depthBuffer: false, generateMipmaps: false,
    minFilter: T.NearestMipmapNearestFilter, magFilter: T.NearestFilter});
  // CubeRenderTarget replaces only attachment zero. Source-valid additional
  // cube attachments are ordinary CubeTexture clones owned by this target.
  target.textures.push(target.texture.clone());
  for (const [index, texture] of target.textures.entries()) {
    texture.isRenderTargetTexture = true;
    texture.type = index ? T.HalfFloatType : T.UnsignedByteType;
    texture.name = index ? 'auxiliary' : 'output';
    texture.mipmaps = Array.from({length: levels[index] - 1}, () => ({}));
  }
  return target;
}
function fixture(t, options = {}) {
  const f = gpuTargetMipmapFixture();
  Object.assign(f.device.limits, {maxTextureArrayLayers: 256, maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  f.device.features = new Set();
  const owner = createThreeRenderTargets(T, {getDevice: () => f.device, flipY: false, ...options});
  t.after(() => owner.dispose());
  return {...f, owner};
}
const passes = f => f.submissions.flatMap(commands => commands.flatMap(command => command.passes));
const selectedView = descriptor => ({dimension: descriptor.dimension, baseArrayLayer: descriptor.baseArrayLayer,
  arrayLayerCount: descriptor.arrayLayerCount, baseMipLevel: descriptor.baseMipLevel, mipLevelCount: descriptor.mipLevelCount});

test('cube MRT preserves all source color identities and matches every pinned backend face/mip view', async t => {
  const f = fixture(t), target = cube(), entry = f.owner.capture(target), backend = new T.WebGPUBackend();
  const context = new RenderContext(); Object.assign(context, {renderTarget: target, textures: target.textures, depth: false});
  for (const texture of target.textures) backend.get(texture).texture = {createView: descriptor => ({...descriptor})};
  assert.deepEqual(entry.shape.colorFormats, ['rgba8unorm', 'rgba16float']);
  assert.deepEqual(entry.shape.colorMipLevelCounts, [4, 3]);
  assert.equal(entry.shape.bytes, 6 * (85 * 4 + 84 * 8));
  const original = target.textures.slice(), images = target.textures.map(texture => texture.image);
  for (let mipLevel = 0; mipLevel < 3; mipLevel++) for (let face = 0; face < 6; face++) {
    context.activeCubeFace = face; context.activeMipmapLevel = mipLevel;
    const expected = backend._getRenderPassDescriptor(context).colorAttachments.map(attachment => selectedView(attachment.view));
    f.owner.render(entry, frame => {
      assert.deepEqual(frame.colorViews.map(view => selectedView(view.descriptor)), expected);
      assert.deepEqual(frame.colorViews.map(view => view.texture.depthOrArrayLayers), [6, 6]);
      assert.deepEqual(frame.colorViews.map(view => view.texture.format), ['rgba8unorm', 'rgba16float']);
    }, {mipLevel, face});
  }
  assert.equal(f.textures.length, 2); assert.equal(f.owner.allocatedBytes, entry.shape.bytes);
  assert.deepEqual(target.textures, original); assert.deepEqual(target.textures.map(texture => texture.image), images);
  assert.ok(target.textures.every(texture => texture.renderTarget === null), 'private owner registration does not patch public pointers');
  const inputs = target.textures.map(texture => f.owner.captureNativeTexture(texture));
  assert.deepEqual(inputs.map(input => input.sampler.lodMaxClamp), [3, 2]);
  assert.ok(inputs.every(input => input.viewDimension === 'cube'));
  f.owner.withNativeTextures(inputs, textures => {
    assert.deepEqual(original.map(texture => textures.get(texture).mipLevelCount), [4, 3]);
    assert.equal(new Set(textures.values()).size, 2);
  });
  assert.equal(f.textures.length, 2, 'native cube sampling borrows the actual colors');
  assert.throws(() => f.owner.render(entry, () => assert.fail('absent secondary mip'), {mipLevel: 3, face: 0}), {code: 'THREE_TARGET_MIP'});
  await f.owner.whenIdle();
});

test('cube MRT generates only captured attachment indices even when the primary cube flag is true', async t => {
  const f = fixture(t), target = cube(), entry = f.owner.capture(target);
  target.texture.generateMipmaps = true;
  let images;
  f.owner.render(entry, frame => { images = frame.colorViews.map(view => view.texture); },
    {face: 5, generateMipmaps: true, generateColorMipmaps: [1]});
  const secondary = passes(f);
  assert.equal(secondary.length, 12, 'two secondary levels, each with six cube faces');
  assert.ok(secondary.every(pass => pass.descriptor.colorAttachments[0].view.texture === images[1]));
  assert.deepEqual(secondary.map(pass => pass.descriptor.colorAttachments[0].view.descriptor.baseArrayLayer),
    [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]);
  for (const pass of secondary) {
    const input = pass.group.entries[0].resource.descriptor, output = pass.descriptor.colorAttachments[0].view.descriptor;
    assert.equal(input.dimension, 'cube'); assert.equal(input.arrayLayerCount, 6);
    assert.equal(input.baseMipLevel + 1, output.baseMipLevel);
  }
  f.owner.render(entry, () => {}, {face: 0, generateMipmaps: true, generateColorMipmaps: []});
  assert.equal(passes(f).length, 12, 'an empty captured selection does not filter any attachment');
  f.owner.render(entry, () => {}, {face: 1, generateColorMipmaps: [0]});
  const primary = passes(f).slice(12); assert.equal(primary.length, 18);
  assert.ok(primary.every(pass => pass.descriptor.colorAttachments[0].view.texture === images[0]));
  await f.owner.whenIdle();
});

test('cube version updates preserve sibling pixels and depth through repeated pyramid replacement', async t => {
  const f = mrtTransferDevice(), owner = createThreeRenderTargets(T, {getDevice: () => f.device, flipY: false});
  t.after(() => owner.dispose());
  const target = cube(); target.depthBuffer = true;
  const first = owner.capture(target); owner.ensure(first);
  const storage = first.storage, [primary, secondary, depth] = f.calls.textures;
  owner.render(first, () => {}, {mipLevel: 2, face: 5});
  const mipDepth = f.calls.textures[3], extraDepthBytes = owner.allocatedBytes - first.shape.bytes;
  for (const [index, texture] of [primary, secondary, depth, mipDepth].entries())
    for (const level of texture.planes) for (const face of level) face.fill(0x41 + index);
  const initialBytes = owner.allocatedBytes;
  f.controls.completion = deferred(); t.after(() => f.controls.completion.resolve());
  target.texture.mipmaps = []; target.texture.generateMipmaps = false;
  target.textures[1].mipmaps = [{}, {}, {}];
  assert.equal(owner.capture(target), first, 'unversioned source edits cannot rebuild either allocated image');
  target.textures[1].needsUpdate = true;
  const next = owner.capture(target); owner.ensure(next);
  assert.notEqual(next, first); assert.deepEqual(next.shape.colorMipLevelCounts, [4, 4]);
  assert.equal(next.storage, storage); assert.equal(first.alive, false);
  assert.throws(() => owner.ensure(first), {code: 'THREE_TARGET_STALE'});
  const replacement = f.calls.textures[4];
  assert.ok(replacement.planes.every(level => level.every(face => face.every(byte => byte === 0))), 'updated source image is reinitialized');
  replacement.planes[2][5].fill(0x75);
  owner.render(next, frame => {
    assert.equal(frame.colorViews[0].texture, primary); assert.equal(frame.colorViews[1].texture, replacement);
    assert.equal(frame.depthView.texture, mipDepth);
    assert.equal(frame.colorViews[1].descriptor.baseArrayLayer, 5);
  }, {mipLevel: 2, face: 5});
  target.texture.needsUpdate = true;
  const smaller = owner.capture(target); owner.ensure(smaller);
  assert.deepEqual(smaller.shape.colorMipLevelCounts, [1, 4], 'primary invalidation cannot collapse the secondary pyramid');
  assert.throws(() => owner.render(smaller, () => {}, {mipLevel: 2}), {code: 'THREE_TARGET_MIP'});
  target.texture.mipmaps = [{}, {}]; target.texture.needsUpdate = true;
  const final = owner.capture(target); owner.ensure(final);
  assert.deepEqual(final.shape.colorMipLevelCounts, [3, 4]);
  owner.render(final, frame => {
    assert.equal(frame.depthView.texture, mipDepth, 'a cached depth image survives shrink and re-expansion');
    assert.equal(frame.colorViews[1].texture, replacement);
  }, {mipLevel: 2, face: 5});
  owner.render(final, frame => assert.equal(frame.depthView.texture, depth));
  assert.ok(primary.planes.every(level => level.every(face => face.every(byte => byte === 0x41))));
  assert.ok(depth.planes[0].every(face => face.every(byte => byte === 0x43)));
  assert.ok(mipDepth.planes[0].every(face => face.every(byte => byte === 0x44)));
  assert.ok(replacement.planes[2][5].every(byte => byte === 0x75));
  assert.equal(owner.allocatedBytes, initialBytes + next.shape.colors[1].colorBytes +
    smaller.shape.colors[0].colorBytes + final.shape.colors[0].colorBytes, 'retired colors remain charged until the GPU fence');
  assert.ok(f.calls.textures.every(texture => texture.destroyed === 0));
  f.controls.completion.resolve(); await owner.whenIdle();
  assert.equal(owner.allocatedBytes, final.shape.bytes + extraDepthBytes);
  assert.deepEqual(f.calls.textures.map(texture => texture.destroyed), [1, 1, 0, 0, 0, 1, 0]);
});

test('cube MRT validates extents, attachment limits and six-face budgets before allocating', t => {
  const f = fixture(t), mismatched = cube();
  mismatched.textures[1].image = Array.from({length: 6}, () => ({width: 4, height: 4, depth: 1}));
  assert.throws(() => f.owner.capture(mismatched), {code: 'THREE_TARGET_SIZE'});
  const source = cube(), shape = inspectThreeRenderTarget(source, T);
  const limited = fixture(t, {maxBytes: shape.bytes - 1}), entry = limited.owner.capture(source);
  assert.throws(() => limited.owner.ensure(entry), {code: 'THREE_TARGET_BUDGET'});
  f.device.limits.maxColorAttachments = 1;
  assert.throws(() => f.owner.capture(source), {code: 'GPU_TARGET_LIMIT'});
  assert.equal(f.textures.length, 0); assert.equal(limited.textures.length, 0);
  const incomplete = new T.CubeRenderTarget(8, {count: 2, depthBuffer: false});
  assert.throws(() => inspectThreeRenderTarget(incomplete, T), {code: 'THREE_TARGET_SOURCE'},
    'the source constructor leaves secondary 2D textures; only complete cube attachments are admitted');
});

test('cube replacement rejects peak-budget and incompatible layout changes before losing initialized content', t => {
  const target = cube(), shape = inspectThreeRenderTarget(target, T);
  const f = fixture(t, {maxBytes: shape.bytes + 6 * 85 * 8 - 1}), entry = f.owner.capture(target);
  f.owner.ensure(entry); const storage = entry.storage, images = f.textures.slice();
  target.textures[1].mipmaps.push({}); target.textures[1].needsUpdate = true;
  assert.throws(() => f.owner.capture(target), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(entry.storage, storage); assert.equal(entry.alive, true);
  assert.deepEqual(storage.diagnostics.colorMipLevelCounts, [4, 3]);
  assert.equal(f.owner.allocatedBytes, shape.bytes); assert.equal(f.textures.length, 2);
  target.textures[1].type = T.FloatType;
  assert.throws(() => f.owner.capture(target), {code: 'THREE_TARGET_UNSUPPORTED'});
  assert.ok(images.every(texture => texture.destroyed === 0));
  assert.equal(entry.storage, storage); assert.equal(f.owner.allocatedBytes, shape.bytes);
});

test('cube generation handoff cannot reenter a native attachment borrow', async t => {
  const f = fixture(t), target = cube(), first = f.owner.capture(target); f.owner.ensure(first);
  const storage = first.storage;
  storage.withTexture(() => {
    target.textures[1].needsUpdate = true;
    assert.throws(() => f.owner.capture(target), {code: 'GPU_TARGET_REENTRANT'});
    assert.equal(first.storage, storage); assert.equal(first.alive, true); assert.equal(f.textures.length, 2);
  });
  const next = f.owner.capture(target); f.owner.ensure(next);
  assert.equal(next.storage, storage); assert.equal(first.alive, false);
  await f.owner.whenIdle(); assert.equal(f.owner.allocatedBytes, next.shape.bytes);
});

test('cube replacement releases reused mip depth exactly once after a view failure', async t => {
  const f = fixture(t), target = createGpuRenderTarget(f.device, {width: 4, height: 4, cube: true,
    colorFormats: ['rgba8unorm', 'rgba16float'], colorMipLevelCounts: [3, 3]});
  t.after(() => target.dispose());
  let depth;
  target.withFrame(frame => { depth = frame.depthView.texture; }, {mipLevel: 1, face: 3});
  target.reinitializeCubeColors([1], [3, 1]); target.reinitializeCubeColors([1], [3, 3]);
  const error = new Error('cached cube depth view failed'); depth.createView = () => { throw error; };
  assert.throws(() => target.withFrame(() => {}, {mipLevel: 1, face: 3}), cause => cause === error);
  await assert.rejects(target.whenIdle(), cause => cause === error);
  assert.ok(f.textures.every(texture => texture.destroyed === 1)); assert.equal(target.allocatedBytes, 0);
});

test('float32 cube mip generation checks the selected secondary format without blocking other colors', async t => {
  const f = fixture(t);
  const target = createGpuRenderTarget(f.device, {width: 4, height: 4, cube: true, depthFormat: null,
    colorFormats: ['rgba8unorm', 'rgba32float'], colorMipLevelCounts: [3, 3]});
  t.after(() => target.dispose()); target.withFrame(() => {}, {face: 5});
  target.generateMipmaps([0]); assert.equal(passes(f).length, 12);
  assert.throws(() => target.generateMipmaps([1]), {code: 'GPU_TARGET_FEATURE'});
  assert.equal(passes(f).length, 12); assert.equal(f.textures[1].format, 'rgba32float');
  f.device.features.add('float32-filterable'); target.generateMipmaps([1]);
  assert.equal(passes(f).length, 24);
  assert.ok(passes(f).slice(12).every(pass => pass.descriptor.colorAttachments[0].view.texture.format === 'rgba32float'));
  await target.whenIdle();
});

test('cube mip feature checks ignore selected colors with no pyramid work', async t => {
  for (const format of ['rgba32float', 'rgba32uint']) {
    const f = fixture(t), target = createGpuRenderTarget(f.device, {width: 4, height: 4, cube: true, depthFormat: null,
      colorFormats: ['rgba8unorm', format], colorMipLevelCounts: [3, 1]});
    t.after(() => target.dispose());
    assert.equal(target.generateMipmaps(), true); assert.equal(passes(f).length, 12);
    assert.equal(target.generateMipmaps(), false, 'a completed pyramid and one-level attachment need no filtering');
    await target.whenIdle();
  }
});

test('public cube MRT readback preserves each attachment, face, mip and raw precision', async t => {
  const f = mrtTransferDevice(), Base = createWebGPURendererClass(T);
  class Renderer extends Base { async _init() { this._device = f.device; this._initialized = true; return this; } }
  const renderer = new Renderer({canvas: {width: 8, height: 8, style: {}}});
  t.after(() => renderer.dispose()); await renderer.init();
  const target = cube(4, [3, 2]), entry = renderer._targets.capture(target);
  renderer.setRenderTarget(target, 4, 1);
  for (let colorIndex = 0; colorIndex < 2; colorIndex++) {
    const [Type, components] = layouts.get(entry.shape.colorFormats[colorIndex]);
    for (let level = 0; level < entry.shape.colorMipLevelCounts[colorIndex]; level++) for (let face = 0; face < 6; face++) {
      const extent = 4 >> level, expected = new Type(extent * extent * components);
      for (let i = 0; i < expected.length; i++) expected[i] = colorIndex
        ? [0xc000, 0x4100, 0x3c00, 0xbc00][(i + face + level) % 4] : (i + face * 31 + level * 17) % 256;
      renderer._targets.ensure(entry);
      entry.storage.withTexture(texture => texture.planes[level][face].set(bytes(expected)), {colorIndex, mipLevel: level});
      const actual = await renderer.readRenderTargetMipPixelsAsync(target, level, {face, colorIndex});
      assert.ok(actual instanceof Type); assert.deepEqual(bytes(actual), bytes(expected));
      assert.equal(f.reads.at(-1).source.mipLevel, level); assert.equal(f.reads.at(-1).source.origin.z, face);
    }
  }
  assert.equal(renderer.getActiveCubeFace(), 4); assert.equal(renderer.getActiveMipmapLevel(), 1);
  assert.equal(f.calls.textures.length, 2);
});

function rendererFixture(t) {
  const device = geometryDevice(); device.textures = []; device.features = new Set();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000,
    maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor});
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  let configuration;
  const canvas = {width: 16, height: 16, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [16, 16], format: configuration.format, usage: 16}); }
  })};
  const Renderer = createWebGPURendererClass(T), renderer = new Renderer({canvas, device});
  t.after(() => renderer.dispose()); return {device, renderer};
}

test('CubeCamera captures every MRT face and both cube outputs feed native source node materials', async t => {
  const {device, renderer} = rendererFixture(t); await renderer.init();
  const N = T.TSL, target = cube(4, [3, 2]), camera = new T.CubeCamera(.1, 10, target), scene = new T.Scene();
  const material = new T.MeshBasicNodeMaterial({side: T.BackSide}); material.colorNode = N.vec3(.25, .5, .75);
  scene.add(new T.Mesh(new T.BoxGeometry(4, 4, 4), material));
  renderer.setMRT(N.mrt({output: N.output, auxiliary: N.vec4(.5, .25, 1.5, 1)}));
  for (const level of [1, 0]) { camera.activeMipmapLevel = level; camera.update(renderer, scene); }
  const destination = new T.RenderTarget(4, 4, {depthBuffer: false}); t.after(() => destination.dispose());
  renderer.setMRT(null); renderer.setRenderTarget(destination);
  const consumer = new T.MeshBasicNodeMaterial();
  consumer.colorNode = N.cubeTexture(target.texture, N.vec3(1, 0, 0)).level(1)
    .add(N.cubeTexture(target.textures[1], N.vec3(0, 1, 0)).level(1));
  new T.QuadMesh(consumer).render(renderer); await renderer.waitForGPU();
  const images = device.textures.filter(texture => texture.depthOrArrayLayers === 6);
  assert.deepEqual(images.map(texture => texture.format), ['rgba8unorm', 'rgba16float']);
  assert.deepEqual(images.map(texture => texture.mipLevelCount), [3, 2]);
  const writes = device.passes.filter(pass => pass.desc.colorAttachments[0].view.texture === images[0]);
  assert.equal(writes.length, 12);
  assert.deepEqual(writes.map(pass => pass.desc.colorAttachments[0].view.baseMipLevel), [...Array(6).fill(1), ...Array(6).fill(0)]);
  assert.deepEqual(writes.map(pass => pass.desc.colorAttachments[1].view.baseArrayLayer), [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]);
  assert.ok(writes.every(pass => pass.desc.colorAttachments[1].view.texture === images[1]));
  const read = device.snapshots.flat().at(-1), resources = [...read.groups.values()]
    .flatMap(({group}) => group.entries.map(entry => entry.resource)).filter(resource => resource?.dimension === 'cube');
  assert.deepEqual(resources.map(resource => resource.texture), images);
  assert.equal(renderer._targets.capture(target).references, 0); assert.equal(renderer._nodeCaptureBytes, 0);
  target.dispose(); assert.ok(images.every(image => image.destroyed));
});
