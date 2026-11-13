/** Cube history ownership and retained source lifecycle; no WGSL/pixel claim. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {createThreeRenderTargets} from './three_render_targets.mjs';
import {inspectThreeNativePass} from './three_node_passes.mjs';
import {gpuTargetMipmapFixture, deferred} from './fixtures/animation/gpu_target_mipmap_fixture.mjs';

function fixture(options = {}) {
  const f = gpuTargetMipmapFixture();
  Object.assign(f.device.limits, {maxTextureArrayLayers: 256, maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  f.device.features = new Set();
  return {...f, owner: createThreeRenderTargets(T, {getDevice: () => f.device, flipY: false, ...options})};
}
const rawCube = (f, options = {}) => createGpuRenderTarget(f.device, {
  width: 8, height: 8, cube: true, format: 'rgba16float', copyDestination: true, ...options,
});
const gone = textures => textures.forEach(texture => assert.equal(texture.destroyed, 1));
const image = frame => frame.colorView.texture;

class CubeHistoryPass extends T.PassNode {
  constructor(size = 8, options = {}) {
    super(T.PassNode.COLOR, new T.Scene(), new T.PerspectiveCamera(), {samples: 0, depthBuffer: false});
    this.renderTarget = new T.CubeRenderTarget(size, {type: T.HalfFloatType, ...options});
    this.renderTarget.texture.name = 'output';
    this._textures = {output: this.renderTarget.texture};
  }
  setSize(width, height) {
    // CubeRenderTarget inherits the ordinary target's logical resize. A cube
    // producer owns its square face descriptors, as in the pinned source.
    assert.equal(width, height);
    for (const face of this.renderTarget.texture.image) Object.assign(face, {width, height});
    this.renderTarget.setSize(width, height);
  }
}
const renderer = f => ({_targets: f.owner, _device: f.device, xr: {isPresenting: false}});

test('cube histories select all six faces and every authored mip without changing shared depth', async () => {
  const f = fixture(), target = rawCube(f, {mipLevelCount: 4});
  target.reserveColorHistory([0]);
  const physical = new Set(), depth = new Map();
  for (const slot of [0, 1, 0]) for (let mipLevel = 0; mipLevel < 4; mipLevel++) for (let face = 0; face < 6; face++) {
    target.withFrame(frame => {
      physical.add(image(frame));
      assert.equal(image(frame).depthOrArrayLayers, 6);
      assert.deepEqual(frame.colorView.descriptor, {baseMipLevel: mipLevel, mipLevelCount: 1,
        dimension: '2d', baseArrayLayer: face, arrayLayerCount: 1});
      const key = mipLevel + '/' + face;
      if (depth.has(key)) assert.equal(frame.depthView, depth.get(key));
      else depth.set(key, frame.depthView);
    }, {colorSlots: [slot], face, mipLevel});
  }
  assert.equal(physical.size, 2);
  assert.equal(target.allocatedBytes, (64 + 16 + 4 + 1) * 6 * (8 * 2 + 4));
  target.withAttachments((colors, depthTexture, histories, withFrame) => {
    assert.notEqual(colors[0], histories[0]);
    for (let face = 0; face < 6; face++) withFrame(frame => {
      assert.equal(image(frame), histories[0]);
      assert.equal(frame.colorView.descriptor.baseArrayLayer, face);
    }, {colorSlots: [1], face, mipLevel: 2});
  });
  assert.throws(() => target.withFrame(() => {}, {face: 6, colorSlots: [1]}), {code: 'GPU_TARGET_FACE'});
  await target.whenIdle(); target.dispose(); f.owner.dispose(); gone(f.textures);
});

test('cube history mip generation uses each complete physical cube and caches its own views', async () => {
  const f = fixture(), target = rawCube(f, {generateMipmaps: true, depthFormat: null});
  target.reserveColorHistory([0]);
  const selected = [];
  for (const slot of [0, 1]) {
    for (let face = 0; face < 6; face++) target.withFrame(() => {}, {face, colorSlots: [slot]});
    target.withTexture(texture => selected.push(texture), {colorSlots: [slot]});
    const passes = f.submissions.at(-1)[0].passes;
    assert.equal(passes.length, 18);
    for (const [index, pass] of passes.entries()) {
      const input = pass.group.entries[0].resource, output = pass.descriptor.colorAttachments[0].view;
      assert.equal(input.texture, selected[slot]); assert.equal(output.texture, selected[slot]);
      assert.deepEqual(input.descriptor, {baseMipLevel: Math.floor(index / 6), mipLevelCount: 1,
        dimension: 'cube', baseArrayLayer: 0, arrayLayerCount: 6});
      assert.equal(output.descriptor.baseArrayLayer, index % 6);
    }
  }
  assert.notEqual(selected[0], selected[1]);
  assert.equal(f.pipelines.length, 1); assert.equal(f.groups.length, 6);
  const submissions = f.submissions.length;
  target.withTexture(() => {}, {colorSlots: [0]}); target.withTexture(() => {}, {colorSlots: [1]});
  assert.equal(f.submissions.length, submissions);
  target.withFrame(() => {}, {face: 4, colorSlots: [1]});
  target.withTexture(() => {}, {colorSlots: [1]});
  assert.equal(f.submissions.length, submissions + 1); assert.equal(f.groups.length, 6);
  assert.equal(target.allocatedBytes, (64 + 16 + 4 + 1) * 6 * 8 * 2);
  await target.whenIdle(); target.dispose(); f.owner.dispose(); gone(f.textures);
});

test('selected cube history copies retain face, mip and physical image across pending resize', async () => {
  const f = fixture(), target = rawCube(f, {mipLevelCount: 3, depthFormat: null});
  target.reserveColorHistory([0]);
  const current = target.selectColorSlots([0]), previous = target.selectColorSlots([1]);
  assert.equal(current.copyFrom(previous, {sourceFace: 1, face: 2, sourceMipLevel: 1,
    destinationMipLevel: 1, depthOrArrayLayers: 3}), true);
  const copy = f.submissions.at(-1)[0].copies[0];
  assert.notEqual(copy.source.texture, copy.destination.texture);
  assert.equal(copy.source.mipLevel, 1); assert.equal(copy.destination.mipLevel, 1);
  assert.equal(copy.source.origin.z, 1); assert.equal(copy.destination.origin.z, 2);
  assert.deepEqual(copy.size, {width: 4, height: 4, depthOrArrayLayers: 3});
  assert.throws(() => current.copyFrom(current, {sourceFace: 2, face: 2}), {code: 'GPU_TARGET_FEEDBACK'});
  assert.equal(current.copyFrom(current, {sourceFace: 1, face: 4}), true);
  const old = f.textures.slice(), fence = deferred(); f.controls.completion = fence;
  target.resize(16, 16);
  assert.ok(old.every(texture => texture.destroyed === 0));
  previous.withFrame(frame => { assert.equal(image(frame).width, 16); }, {face: 5, mipLevel: 2});
  fence.resolve(); await target.whenIdle(); gone(old);
  target.dispose(); f.owner.dispose(); gone(f.textures);
});

test('cube history budget rejection counts six full pyramids before allocating an alternate', async () => {
  const f = fixture(), bytes = (64 + 16 + 4 + 1) * 6 * 8;
  const target = rawCube(f, {mipLevelCount: 4, depthFormat: null, maxBytes: bytes * 2 - 1});
  const count = f.textures.length;
  assert.throws(() => target.reserveColorHistory([0]), {code: 'GPU_TARGET_BUDGET'});
  assert.equal(f.textures.length, count); assert.equal(target.allocatedBytes, bytes);
  target.withFrame(frame => assert.equal(image(frame).depthOrArrayLayers, 6), {face: 5});
  await target.whenIdle(); target.dispose(); f.owner.dispose(); gone(f.textures);
});

test('retained PassNode cube cloning and swaps preserve private ownership and every face', async () => {
  const f = fixture(), pass = new CubeHistoryPass(), original = pass.renderTarget.texture;
  pass.getPreviousTextureNode();
  const alternate = pass.getPreviousTexture('output');
  assert.notEqual(original.version, alternate.version, 'source clone has an independent version');
  assert.equal(inspectThreeNativePass(T, renderer(f), pass), true);
  const originals = [];
  let originalTexture;
  for (let face = 0; face < 6; face++) f.owner.render(f.owner.capture(pass.renderTarget), frame => {
    originals.push(frame); originalTexture = image(frame);
  }, {face});
  for (let step = 0; step < 4; step++) {
    pass.toggleTexture('output'); f.owner.registerNativePass(pass);
    const entry = f.owner.capture(pass.renderTarget), before = f.owner.captureNativeTexture(pass.getPreviousTexture('output'));
    const current = f.owner.captureNativeTexture(pass.getTexture('output'));
    assert.equal(current.viewDimension, 'cube'); assert.equal(before.viewDimension, 'cube');
    for (let face = 0; face < 6; face++) f.owner.render(entry, frame => {
      assert.equal(frame.colorView.descriptor.baseArrayLayer, face);
      assert.equal(frame.depthView, originals[face].depthView);
      if (step % 2) assert.equal(image(frame), originalTexture); else assert.notEqual(image(frame), originalTexture);
      f.owner.withNativeTexture(before, texture => assert.notEqual(texture, image(frame)));
      assert.throws(() => f.owner.withNativeTexture(current, () => {}), {code: 'THREE_TARGET_FEEDBACK'});
    }, {face});
    f.owner.withNativeTexture(before, texture => {
      for (const face of [5, 0, 3]) f.owner.render(entry, frame => assert.notEqual(image(frame), texture), {face});
    });
    f.owner.withNativeTexture(current, () => {
      // A cube sampler exposes every face, so even a different face aliases it.
      assert.throws(() => f.owner.render(entry, () => {}, {face: 4}), {code: 'THREE_TARGET_FEEDBACK'});
    });
  }
  assert.equal(original.renderTarget, null); assert.equal(alternate.renderTarget, null);
  assert.equal(pass._textureNodes.output.value, pass.renderTarget.texture);
  assert.equal(pass._previousTextureNodes.output.value, pass.getPreviousTexture('output'));
  assert.equal(f.owner.allocatedBytes, 8 * 8 * 6 * (8 * 2 + 4));
  await f.owner.whenIdle(); f.owner.dispose(); gone(f.textures);
});

test('late cube history preserves initialized mip storage and source generation choices', async () => {
  const f = fixture(), pass = new CubeHistoryPass(8, {generateMipmaps: true, minFilter: T.LinearMipmapLinearFilter});
  const original = pass.renderTarget.texture, first = f.owner.capture(pass.renderTarget);
  let imageBefore;
  f.owner.render(first, frame => { imageBefore = image(frame); }, {face: 5, generateColorMipmaps: [0]});
  pass.getPreviousTextureNode(); f.owner.registerNativePass(pass);
  f.owner.render(f.owner.capture(pass.renderTarget), frame => assert.equal(image(frame), imageBefore), {face: 0});
  pass.toggleTexture('output'); f.owner.registerNativePass(pass);
  pass.renderTarget.texture.generateMipmaps = false;
  const current = f.owner.capture(pass.renderTarget), before = f.owner.captureNativeTexture(original);
  assert.equal(current.shape.mipLevelCount, 4);
  f.owner.withNativeTexture(before, texture => {
    assert.equal(texture, imageBefore);
    f.owner.render(current, frame => {
      assert.notEqual(image(frame), texture);
      assert.equal(frame.colorView.descriptor.baseArrayLayer, 3);
      assert.equal(frame.colorView.descriptor.baseMipLevel, 2);
    }, {face: 3, mipLevel: 2, generateColorMipmaps: []});
  });
  assert.equal(original.renderTarget, null);
  await f.owner.whenIdle(); f.owner.dispose(); gone(f.textures);
});

test('cube pass resize keeps queued generations until readers and GPU work finish; disposal invalidates them', async () => {
  const f = fixture(), pass = new CubeHistoryPass();
  pass.getPreviousTextureNode(); f.owner.registerNativePass(pass);
  const first = f.owner.captureNativeTexture(pass.getTexture('output'));
  const before = f.owner.captureNativeTexture(pass.getPreviousTexture('output'));
  const releaseFirst = f.owner.retainGeneration(first.entry), releaseBefore = f.owner.retainGeneration(before.entry);
  f.owner.withNativeTextures([first, before], () => {});
  const old = f.textures.slice(), fence = deferred(); f.controls.completion = fence;
  f.owner.withNativeEffectTargets([pass.renderTarget], () => pass.setSize(16, 16));
  pass.toggleTexture('output'); f.owner.registerNativePass(pass);
  f.owner.render(f.owner.capture(pass.renderTarget), frame => assert.equal(image(frame).width, 16), {face: 5});
  f.owner.withNativeTextures([first, before], textures => {
    assert.equal(textures.get(first.texture).width, 8);
    assert.equal(textures.get(before.texture).depthOrArrayLayers, 6);
  });
  releaseFirst(); releaseBefore();
  assert.ok(old.every(texture => texture.destroyed === 0));
  fence.resolve(); await f.owner.whenIdle(); gone(old);
  const next = f.owner.captureNativeTexture(pass.getTexture('output'));
  pass.renderTarget.dispose();
  assert.throws(() => f.owner.withNativeTexture(next, () => {}), {code: 'THREE_TARGET_STALE'});
  f.owner.dispose(); gone(f.textures);
});

test('cube history admission rejects foreign owners and mismatched storage before allocation', () => {
  const f = fixture(), pass = new CubeHistoryPass();
  pass.getPreviousTextureNode();
  const previous = pass.getPreviousTexture('output'), foreign = new T.CubeRenderTarget(8);
  previous.renderTarget = foreign;
  assert.throws(() => inspectThreeNativePass(T, renderer(f), pass), {code: 'THREE_NODE_PROGRAM_PASS'});
  previous.renderTarget = null; previous.type = T.UnsignedByteType;
  assert.throws(() => f.owner.registerNativePass(pass), {code: 'THREE_TARGET_SOURCE'});
  assert.equal(f.textures.length, 0);
  f.owner.dispose();
});


test('registered previous cube textures cannot be adopted by another target', () => {
  const f = fixture(), pass = new CubeHistoryPass();
  pass.getPreviousTextureNode(); f.owner.registerNativePass(pass);
  const previous = pass.getPreviousTexture('output'), foreign = new T.CubeRenderTarget(8);
  foreign.texture = previous;
  assert.throws(() => f.owner.capture(foreign), {code: 'THREE_TARGET_SOURCE'});
  assert.equal(f.owner.captureNativeTexture(previous).entry.target, pass.renderTarget);
  assert.equal(previous.renderTarget, null); assert.equal(f.textures.length, 0);
  f.owner.dispose();
});
