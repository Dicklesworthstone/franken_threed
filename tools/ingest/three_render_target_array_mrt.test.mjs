/** Recorded owner calls and descriptors, not source-class or rasterized parity. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {createThreeRenderTargets, inspectThreeRenderTarget, inspectThreeRenderTargetMip} from './three_render_targets.mjs';
import {targetThree} from './fixtures/target_residency_fixture.mjs';
import {gpuTargetMipmapFixture, deferred} from './fixtures/animation/gpu_target_mipmap_fixture.mjs';

const T = {...targetThree, NearestMipmapNearestFilter: 1004};
function fixture(options = {}) {
  const f = gpuTargetMipmapFixture();
  Object.assign(f.device.limits, {maxTextureArrayLayers: 256, maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  f.device.features = new Set();
  return {...f, owner: createThreeRenderTargets(T, {getDevice: () => f.device, flipY: false, ...options})};
}
function color(target, index) {
  const texture = new T.Texture();
  Object.assign(texture, {isRenderTargetTexture: true, renderTarget: target,
    image: {width: target.width, height: target.height, depth: target.depth},
    type: index === 0 ? T.UnsignedByteType : T.HalfFloatType,
    minFilter: T.NearestMipmapNearestFilter, magFilter: T.NearestFilter,
    generateMipmaps: index === 0, mipmaps: index === 0 ? [] : [{}, {}, {}]});
  return texture;
}
function target(samples = 0) {
  const target = new T.RenderTarget(8, 4, {depth: 3, depthBuffer: false, samples});
  target.textures = [color(target, 0), color(target, 1)];
  return target;
}
function resize(target, width, height) {
  target.width = width; target.height = height;
  for (const texture of target.textures) Object.assign(texture.image, {width, height});
  target.dispose();
}
const passes = f => f.submissions.flatMap(commands => commands.flatMap(command => command.passes));
const destroyed = f => f.textures.forEach(texture => assert.equal(texture.destroyed, 1));

test('native array MRT retains independent mip allocations, formats, samplers and every layer', async () => {
  const f = fixture(), source = target(), shape = inspectThreeRenderTarget(source, T);
  assert.deepEqual(shape.colorMipLevelCounts, [4, 3]);
  assert.deepEqual(shape.colorFormats, ['rgba8unorm', 'rgba16float']);
  assert.equal(shape.bytes, 3 * (43 * 4 + 42 * 8));
  assert.equal(shape.viewDimension, '2d-array');
  const inputs = source.textures.map(texture => f.owner.captureNativeTexture(texture));
  assert.deepEqual(inputs.map(input => input.sampler.lodMaxClamp), [3, 2]);
  assert.ok(inputs.every(input => input.viewDimension === '2d-array'));
  f.owner.withNativeTextures(inputs, textures => {
    const images = source.textures.map(texture => textures.get(texture));
    assert.deepEqual(images.map(image => image.mipLevelCount), [4, 3]);
    assert.deepEqual(images.map(image => image.depthOrArrayLayers), [3, 3]);
    assert.deepEqual(images.map(image => image.format), ['rgba8unorm', 'rgba16float']);
  });
  assert.equal(f.owner.allocatedBytes, shape.bytes);
  assert.equal(f.textures.length, 2, 'native reads need no orientation replica');
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f);
});

test('array MRT render admission uses the intersection of color mips and preserves the selected layer', async () => {
  const f = fixture(), source = target(), entry = f.owner.capture(source);
  assert.throws(() => f.owner.render(entry, () => assert.fail('missing secondary mip'), {mipLevel: 3, face: 2}),
    {code: 'THREE_TARGET_MIP'});
  assert.throws(() => f.owner.render(entry, () => assert.fail('missing array layer'), {mipLevel: 2, face: 3}),
    {code: 'THREE_TARGET_FACE'});
  assert.equal(f.textures.length, 0, 'invalid destinations must not allocate');
  assert.deepEqual(inspectThreeRenderTargetMip(entry.shape, 2, 2), {width: 2, height: 1});
  let selected;
  f.owner.render(entry, frame => { selected = frame.colorViews; }, {mipLevel: 2, face: 2});
  assert.deepEqual(selected.map(view => view.descriptor.baseMipLevel), [2, 2]);
  assert.deepEqual(selected.map(view => view.descriptor.baseArrayLayer), [2, 2]);
  assert.ok(selected.every(view => view.descriptor.mipLevelCount === 1 && view.descriptor.arrayLayerCount === 1));
  f.owner.render(entry, frame => assert.notEqual(frame.colorViews[0], selected[0]), {mipLevel: 1, face: 0});
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f);
});

test('selected array MRT generation filters only that color across all layers after its render', async () => {
  const f = fixture(), source = target(), entry = f.owner.capture(source);
  let images;
  f.owner.render(entry, frame => { images = frame.colorViews.map(view => view.texture); },
    {face: 2, generateColorMipmaps: [1]});
  const secondary = passes(f);
  assert.equal(secondary.length, 6);
  assert.ok(secondary.every(pass => pass.descriptor.colorAttachments[0].view.texture === images[1]));
  assert.deepEqual(secondary.map(pass => pass.descriptor.colorAttachments[0].view.descriptor.baseArrayLayer), [0, 1, 2, 0, 1, 2]);
  assert.deepEqual(secondary.map(pass => pass.descriptor.colorAttachments[0].view.descriptor.baseMipLevel), [1, 1, 1, 2, 2, 2]);
  for (const pass of secondary) {
    const input = pass.group.entries[0].resource.descriptor, output = pass.descriptor.colorAttachments[0].view.descriptor;
    assert.equal(input.baseArrayLayer, output.baseArrayLayer);
    assert.equal(input.baseMipLevel + 1, output.baseMipLevel);
  }
  source.texture.generateMipmaps = false; source.textures[1].generateMipmaps = true;
  assert.equal(f.owner.capture(source), entry, 'flag edits do not replace initialized pyramids');
  f.owner.render(entry, () => {}, {face: 0, generateColorMipmaps: []});
  assert.equal(passes(f).length, 6, 'a clear-only/no-generation request keeps lower levels');
  f.owner.render(entry, () => {}, {face: 1, generateColorMipmaps: [0]});
  const primary = passes(f).slice(6);
  assert.equal(primary.length, 9);
  assert.ok(primary.every(pass => pass.descriptor.colorAttachments[0].view.texture === images[0]));
  f.owner.withNativeTextures(source.textures.map(texture => f.owner.captureNativeTexture(texture)), () => {});
  assert.equal(passes(f).length, 15, 'sampling cannot reinterpret later source generation flags');
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f);
});

for (const samples of [0, 4]) test(`array MRT charges all chains and ${samples || 1}-sample layer storage before allocation`, async () => {
  const source = target(samples), bytes = 3 * (43 * 4 + 42 * 8) + (samples === 4 ? 3 * 32 * 12 * 4 : 0);
  const limited = fixture({maxBytes: bytes - 1}), rejected = limited.owner.capture(source);
  assert.throws(() => limited.owner.ensure(rejected), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(limited.textures.length, 0); limited.owner.dispose();
  const f = fixture({maxBytes: bytes}), entry = f.owner.capture(source);
  for (const face of [0, 1, 2]) f.owner.render(entry, frame => {
    if (samples === 4) {
      assert.ok(frame.colorViews.every(view => view.texture.sampleCount === 4 && view.texture.depthOrArrayLayers === 1));
      assert.deepEqual(frame.resolveTargets.map(view => view.descriptor.baseArrayLayer), [face, face]);
    } else assert.deepEqual(frame.colorViews.map(view => view.descriptor.baseArrayLayer), [face, face]);
  }, {face, generateColorMipmaps: [1]});
  assert.equal(f.owner.allocatedBytes, bytes);
  if (samples === 4) assert.throws(() => f.owner.render(entry, () => {}, {mipLevel: 1, face: 2}), {code: 'THREE_TARGET_MIP'});
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f);
});

test('accepted array MRT generations retain complete chains across resize and submission fences', async () => {
  const f = fixture(), source = target(), entry = f.owner.capture(source);
  const inputs = source.textures.map(texture => f.owner.captureNativeTexture(texture)), release = f.owner.retainGeneration(entry);
  f.owner.render(entry, () => {}, {face: 2, generateColorMipmaps: [0, 1]});
  const original = f.textures.slice(), fence = deferred(); f.controls.completion = fence;
  f.owner.withNativeTextures(inputs, () => {});
  f.owner.withNativePassResize(source, 4, 2, () => resize(source, 4, 2));
  const next = f.owner.capture(source);
  assert.notEqual(next, entry);
  f.owner.render(next, frame => assert.ok(frame.colorViews.every(view => view.texture.width === 4)), {face: 1});
  f.owner.withNativeTextures(inputs, textures => assert.ok([...textures.values()].every(texture =>
    texture.width === 8 && texture.depthOrArrayLayers === 3)));
  assert.equal(f.owner.allocatedBytes, entry.shape.bytes + next.shape.bytes);
  release();
  assert.ok(original.every(texture => texture.destroyed === 0));
  fence.resolve(); await f.owner.whenIdle(); await f.owner.whenIdle();
  assert.ok(original.every(texture => texture.destroyed === 1));
  assert.equal(f.owner.allocatedBytes, next.shape.bytes);
  f.owner.dispose(); destroyed(f);
});

test('array MRT histories keep both physical mip pyramids and permit previous-frame layer reads', async () => {
  const f = fixture(), source = target();
  const initial = source.textures.slice(), previous = [color(source, 0), color(source, 1)];
  const pass = {renderTarget: source, _textures: {output: initial[0], normal: initial[1]},
    _previousTextures: {output: previous[0], normal: previous[1]}};
  f.owner.registerNativePass(pass);
  const first = f.owner.capture(source), inputs = initial.map(texture => f.owner.captureNativeTexture(texture));
  let firstImages;
  f.owner.withNativeTextures(inputs, textures => { firstImages = initial.map(texture => textures.get(texture)); });
  pass._textures = {output: previous[0], normal: previous[1]};
  pass._previousTextures = {output: initial[0], normal: initial[1]};
  source.textures = previous.slice(); f.owner.registerNativePass(pass);
  const current = f.owner.capture(source), secondInputs = previous.map(texture => f.owner.captureNativeTexture(texture));
  let secondImages;
  f.owner.withNativeTextures(secondInputs, textures => { secondImages = previous.map(texture => textures.get(texture)); });
  assert.ok(secondImages.every((image, index) => image !== firstImages[index]));
  assert.deepEqual(secondImages.map(image => image.mipLevelCount), [4, 3]);
  assert.equal(f.owner.allocatedBytes, 2 * first.shape.bytes);
  f.owner.withNativeTextures(inputs, textures => {
    assert.equal(textures.get(initial[0]), firstImages[0]);
    f.owner.render(current, frame => {
      assert.deepEqual(frame.colorViews.map(view => view.texture), secondImages);
      assert.deepEqual(frame.colorViews.map(view => view.descriptor.baseArrayLayer), [2, 2]);
      assert.deepEqual(frame.colorViews.map(view => view.descriptor.baseMipLevel), [2, 2]);
    }, {mipLevel: 2, face: 2});
  });
  f.owner.withNativeTextures(inputs, () => f.owner.render(current, () => {}, {face: 1, generateColorMipmaps: [1]}));
  assert.equal(passes(f).length, 6);
  assert.ok(passes(f).every(draw => draw.descriptor.colorAttachments[0].view.texture === secondImages[1]));
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f);
});

test('array MRT attachment disposal invalidates accepted inputs without leaving resident chains', async () => {
  const f = fixture(), source = target(), input = f.owner.captureNativeTexture(source.textures[1]);
  f.owner.withNativeTexture(input, () => {});
  source.textures[1].dispose();
  assert.throws(() => f.owner.withNativeTexture(input, () => {}), {code: 'THREE_TARGET_STALE'});
  await f.owner.whenIdle(); f.owner.dispose();
  assert.equal(f.owner.allocatedBytes, 0); destroyed(f);
});
