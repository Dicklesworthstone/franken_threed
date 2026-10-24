/** Native attachment ownership and pinned-source identities; GPU pixel proof is in the browser parity suite. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {createThreeRenderTargets, inspectThreeRenderTarget} from './three_render_targets.mjs';
import {gpuPostprocessFixture, deferred} from './fixtures/animation/gpu_postprocess_fixture.mjs';

function fixture(options = {}) {
  const f = gpuPostprocessFixture(), create = f.device.createTexture;
  Object.assign(f.device.limits, {maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32});
  f.device.features = new Set();
  f.device.createTexture = descriptor => {
    const texture = create(descriptor);
    texture.sampleCount = descriptor.sampleCount; texture.mipLevelCount = descriptor.mipLevelCount;
    return texture;
  };
  const owner = createThreeRenderTargets(T, {getDevice: () => f.device, ...options});
  return {...f, owner};
}
function target(samples = 0) {
  return new T.RenderTarget(16, 8, {count: 2, type: T.HalfFloatType, samples,
    depthTexture: new T.DepthTexture(16, 8, T.FloatType)});
}
const destroyed = textures => textures.forEach(texture => assert.equal(texture.destroyed, 1));

for (const sampleCount of [1, 4]) test(`MRT shares depth and resolves each of ${sampleCount} color samples independently`, async () => {
  const f = fixture(), formats = ['rgba16float', 'rgba8unorm'];
  const raw = createGpuRenderTarget(f.device, {width: 16, height: 8, colorFormats: formats, sampleCount, sampleDepth: true});
  assert.deepEqual(raw.rendererOptions.colorFormats, formats);
  let frame;
  raw.withFrame(value => { frame = value; });
  assert.equal(frame.colorViews.length, 2); assert.equal(frame.colorView, frame.colorViews[0]);
  assert.deepEqual(frame.colorViews.map(view => view.texture.sampleCount), [sampleCount, sampleCount]);
  assert.equal(frame.depthView.texture.sampleCount, sampleCount);
  raw.withAttachments((colors, depth) => {
    assert.equal(Object.isFrozen(colors), true); assert.equal(colors.length, 2);
    assert.equal(depth, frame.depthView.texture);
    for (let index = 0; index < 2; index++) {
      assert.equal(colors[index].sampleCount, 1); assert.equal(colors[index].format, formats[index]);
      assert.equal(sampleCount === 4 ? frame.resolveTargets[index].texture : frame.colorViews[index].texture, colors[index]);
    }
    assert.throws(() => raw.withTexture(() => {}), {code: 'GPU_TARGET_REENTRANT'});
  });
  raw.withTexture(texture => assert.equal(texture.format, formats[1]), {colorIndex: 1});
  assert.throws(() => raw.withTexture(() => {}, {colorIndex: 2}), {code: 'GPU_TARGET_ATTACHMENT'});
  assert.throws(() => raw.withAttachments(async () => {}), {code: 'GPU_TARGET_ASYNC_USE'});
  assert.equal(raw.allocatedBytes, 16 * 8 * (12 * (sampleCount === 4 ? 5 : 1) + 4 * sampleCount));
  await raw.whenIdle(); raw.dispose(); destroyed(f.calls.textures); f.owner.dispose();
});

test('MRT count, render byte cost, storage budget and unsupported mips reject before allocation', () => {
  const f = fixture();
  const base = {width: 16, height: 8, colorFormats: ['rgba8unorm', 'rgba8unorm']};
  f.device.limits.maxColorAttachments = 1;
  assert.throws(() => createGpuRenderTarget(f.device, base), {code: 'GPU_TARGET_LIMIT'});
  f.device.limits.maxColorAttachments = 8;
  assert.throws(() => createGpuRenderTarget(f.device, {...base, colorFormats: Array(5).fill('rgba8unorm')}), {code: 'GPU_TARGET_LIMIT'});
  assert.throws(() => createGpuRenderTarget(f.device, {...base, maxBytes: 16 * 8 * 12 - 1}), {code: 'GPU_TARGET_BUDGET'});
  assert.throws(() => createGpuRenderTarget(f.device, {...base, generateMipmaps: true}), {code: 'GPU_TARGET_UNSUPPORTED'});
  const source = target(); source.textures.push(source.texture);
  assert.throws(() => f.owner.capture(source), {code: 'THREE_TARGET_SOURCE'});
  assert.equal(f.calls.textures.length, 0); f.owner.dispose();
});

test('MRT retirement and partial allocation failure release every color and shared depth exactly once', async () => {
  const f = fixture(), raw = createGpuRenderTarget(f.device, {width: 16, height: 8,
    colorFormats: ['rgba8unorm', 'rgba16float'], sampleCount: 4});
  const first = f.calls.textures.slice(), fence = deferred(); f.controls.completion = fence;
  raw.withAttachments(() => {}); raw.resize(8, 4);
  first.forEach(texture => assert.equal(texture.destroyed, 0));
  fence.resolve(); await raw.whenIdle(); destroyed(first);
  raw.dispose(); destroyed(f.calls.textures); f.owner.dispose();
  for (const failure of [2, 3, 4, 5]) {
    const failed = fixture(); failed.controls.textureFailure = failure;
    assert.throws(() => createGpuRenderTarget(failed.device, {colorFormats: ['rgba8unorm', 'rgba16float'], sampleCount: 4}), /injected/);
    destroyed(failed.calls.textures); failed.owner.dispose();
  }
});

test('pinned PassNode named attachments keep source identities and borrow distinct native color images', async () => {
  const f = fixture(), pass = T.TSL.pass(new T.Scene(), new T.PerspectiveCamera(), {samples: 0});
  const normal = pass.getTextureNode('normal').value, source = pass.renderTarget;
  const shape = inspectThreeRenderTarget(source, T);
  assert.deepEqual(shape.textures, source.textures); assert.equal(shape.colors[1].texture, normal);
  assert.equal(normal.renderTarget, source); assert.deepEqual(shape.colorFormats, ['rgba16float', 'rgba16float']);
  const snapshots = [...source.textures, source.depthTexture].map(texture => f.owner.captureNativeTexture(texture));
  let frame;
  f.owner.render(snapshots[0].entry, value => {
    frame = value;
    assert.throws(() => f.owner.withNativeTextures(snapshots, () => {}), {code: 'THREE_TARGET_REENTRANT'});
  });
  f.owner.withNativeTextures(snapshots, textures => {
    assert.equal(textures.size, 3);
    assert.equal(textures.get(normal), frame.colorViews[1].texture);
    assert.equal(textures.get(source.texture), frame.colorViews[0].texture);
    assert.equal(textures.get(source.depthTexture), frame.depthView.texture);
    assert.equal(snapshots[0].entry.sample, null);
  });
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f.calls.textures);
});

test('each native MRT color retains its own precision and sampler type', async () => {
  const f = fixture(), source = target();
  const second = source.textures[1]; second.type = T.FloatType; second.format = T.RedFormat;
  second.minFilter = second.magFilter = T.NearestFilter;
  const inputs = source.textures.map(texture => f.owner.captureNativeTexture(texture));
  assert.deepEqual(inputs.map(input => input.sampleType), ['float', 'unfilterable-float']);
  f.owner.withNativeTextures(inputs, textures => {
    assert.equal(textures.get(source.texture).format, 'rgba16float');
    assert.equal(textures.get(second).format, 'r32float');
  });
  assert.equal(f.owner.allocatedBytes, 16 * 8 * (8 + 4 + 4));
  await f.owner.whenIdle(); f.owner.dispose(); destroyed(f.calls.textures);
});

test('MRT automatic resize preserves every accepted attachment generation until its leases complete', async () => {
  const f = fixture(), source = target(4), inputs = source.textures.map(texture => f.owner.captureNativeTexture(texture));
  const entry = inputs[0].entry, release = f.owner.retainGeneration(entry);
  f.owner.render(entry, () => {}); const before = f.calls.textures.slice();
  f.owner.withNativePassResize(source, 32, 16, () => source.setSize(32, 16));
  f.owner.withNativeTextures(inputs, textures => {
    for (const texture of textures.values()) assert.equal(texture.width, 16);
  });
  const next = f.owner.capture(source); assert.notEqual(next, entry);
  f.owner.render(next, frame => frame.colorViews.forEach(view => assert.equal(view.texture.width, 32)));
  before.forEach(texture => assert.equal(texture.destroyed, 0));
  release(); await f.owner.whenIdle(); destroyed(before);
  f.owner.dispose(); destroyed(f.calls.textures);
});

test('secondary attachment replacement and disposal invalidate captured target generations', async () => {
  for (const replace of [false, true]) {
    const f = fixture(), source = target(), input = f.owner.captureNativeTexture(source.textures[1]);
    f.owner.withNativeTexture(input, () => {});
    if (replace) source.textures[1] = source.textures[1].clone(); else source.textures[1].dispose();
    assert.throws(() => f.owner.withNativeTexture(input, () => {}), {code: 'THREE_TARGET_STALE'});
    if (replace) assert.notEqual(f.owner.capture(source), input.entry);
    f.owner.dispose(); destroyed(f.calls.textures);
  }
});
