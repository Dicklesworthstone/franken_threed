import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {gpuTargetMipmapFixture, deferred} from './fixtures/animation/gpu_target_mipmap_fixture.mjs';

function pyramid(width, height) {
  const result = [[width, height]];
  while (width > 1 || height > 1) {
    width = Math.max(1, Math.floor(width / 2)); height = Math.max(1, Math.floor(height / 2));
    result.push([width, height]);
  }
  return result;
}
const make = (f, options = {}) => createGpuRenderTarget(f.device,
  {width: 8, height: 4, format: 'rgba8unorm', depthFormat: null, generateMipmaps: true, ...options});
const gone = f => { for (const t of f.textures) assert.equal(t.destroyed, 1); };

for (const [width, height] of [[8, 4], [7, 5], [1, 9], [9, 1], [1, 1]]) {
  for (const format of ['rgba8unorm', 'rgba8unorm-srgb', 'bgra8unorm', 'bgra8unorm-srgb', 'rgba16float']) {
    test(`${width}x${height} ${format}: full bounded pyramid and disjoint mip render passes`, async () => {
      const f = gpuTargetMipmapFixture(), t = make(f, {width, height, format}), shape = pyramid(width, height);
      const bytes = shape.reduce((n, [w, h]) => n + w * h, 0) * (format === 'rgba16float' ? 8 : 4);
      assert.equal(t.mipLevelCount, shape.length); assert.equal(t.allocatedBytes, bytes);
      assert.equal(f.submissions.length, 0, 'generation is lazy');
      t.withFrame((frame, texture) => {
        assert.equal(frame.colorView.descriptor.baseMipLevel, 0);
        assert.equal(frame.colorView.descriptor.mipLevelCount, 1);
        assert.equal(texture.mipLevelCount, shape.length);
      });
      t.withTexture(texture => {
        assert.equal(texture, f.textures[0]);
        assert.equal(texture.createView().descriptor.mipLevelCount, shape.length);
        assert.equal(f.submissions.length, shape.length > 1 ? 1 : 0, 'generation precedes consumer');
      });
      const passes = f.submissions[0]?.[0].passes ?? [];
      assert.equal(passes.length, shape.length - 1);
      for (let i = 0; i < passes.length; i++) {
        const pass = passes[i], source = pass.group.entries[0].resource;
        assert.equal(source.descriptor.baseMipLevel, i);
        assert.equal(source.descriptor.mipLevelCount, 1);
        assert.equal(pass.descriptor.colorAttachments[0].view.descriptor.baseMipLevel, i + 1);
        assert.equal(pass.pipeline.descriptor.fragment.targets[0].format, format);
        assert.equal(pass.vertices, 3); assert.ok(pass.ended);
      }
      await t.whenIdle(); assert.deepEqual(f.scopes, []); t.dispose(); gone(f);
    });
  }
}

test('read reuse does not regenerate; each submitted write invalidates the same cached pyramid', () => {
  const f = gpuTargetMipmapFixture(), t = make(f);
  t.withTexture(() => {});
  const groups = f.groups.length;
  t.withTexture(() => {}); t.withTexture(() => {});
  assert.equal(f.submissions.length, 1); assert.equal(t.diagnostics.mipGenerations, 1);
  t.withFrame(() => {}); t.withTexture(() => {});
  assert.equal(f.submissions.length, 2); assert.equal(t.diagnostics.mipGenerations, 2);
  assert.equal(f.groups.length, groups); assert.equal(f.pipelines.length, 1); assert.equal(f.samplers.length, 1);
  assert.throws(() => t.withFrame(() => { throw Error('caller submitted then failed'); }), /caller/);
  t.withTexture(() => {}); assert.equal(t.diagnostics.mipGenerations, 3);
  t.dispose(); gone(f);
});

test('MSAA resolves into level zero, generates only resolved color, and accounts all samples', () => {
  const f = gpuTargetMipmapFixture(), t = make(f, {sampleCount: 4, depthFormat: 'depth32float'});
  const [color, msaa, depth] = f.textures;
  assert.equal(color.mipLevelCount, 4); assert.equal(msaa.mipLevelCount, 1); assert.equal(depth.mipLevelCount, 1);
  assert.equal(t.allocatedBytes, (32 + 8 + 2 + 1) * 4 + 32 * 4 * 4 + 32 * 4 * 4);
  t.withFrame(frame => {
    assert.equal(frame.colorView.texture, msaa); assert.equal(frame.resolveTarget.texture, color);
    assert.equal(frame.resolveTarget.descriptor.mipLevelCount, 1);
  });
  t.withTexture(() => {});
  for (const pass of f.submissions[0][0].passes) assert.equal(pass.group.entries[0].resource.texture, color);
  t.dispose(); gone(f);
});

test('disabled generation preserves a single mip and needs none of the pipeline APIs', () => {
  const f = gpuTargetMipmapFixture(); delete f.device.createRenderPipeline; delete f.device.createSampler;
  const t = make(f, {generateMipmaps: false});
  t.withFrame(() => {}); t.withTexture(() => {});
  assert.equal(t.mipLevelCount, 1); assert.equal(t.allocatedBytes, 128);
  assert.equal(f.submissions.length, 0); t.dispose(); gone(f);
});

test('full mip budget rejects before allocation and invalid opt-in values are refused', () => {
  const f = gpuTargetMipmapFixture();
  assert.throws(() => make(f, {maxBytes: 171}), {code: 'GPU_TARGET_BUDGET'});
  for (const value of [1, 'true', null, {}]) assert.throws(() => make(f, {generateMipmaps: value}), {code: 'GPU_TARGET_OPTIONS'});
  assert.equal(f.textures.length, 0);
  const t = make(f, {maxBytes: 172}); assert.equal(t.allocatedBytes, 172); t.dispose();
});

test('resize keeps old mip resources fenced, rebuilds the chain, and charges peak bytes', async () => {
  const f = gpuTargetMipmapFixture(), t = make(f), fence = deferred();
  t.withTexture(() => {}); const old = f.textures[0]; f.controls.completion = fence;
  t.resize(7, 3); assert.equal(old.destroyed, 0); assert.equal(t.mipLevelCount, 3);
  assert.equal(t.allocatedBytes, 172 + (21 + 3 + 1) * 4);
  t.withTexture(() => {});
  assert.equal(f.submissions[1][0].passes.length, 2);
  for (const pass of f.submissions[1][0].passes) assert.notEqual(pass.group.entries[0].resource.texture, old);
  assert.equal(f.pipelines.length, 1);
  fence.resolve(); await t.whenIdle(); assert.equal(old.destroyed, 1); assert.equal(t.allocatedBytes, 100);
  t.dispose(); gone(f);
});

test('mip cost participates in allocation-free resize rejection', () => {
  const f = gpuTargetMipmapFixture(), t = make(f, {maxBytes: 250});
  t.withTexture(() => {});
  assert.throws(() => t.resize(7, 3), {code: 'GPU_TARGET_BUDGET'});
  assert.equal(f.textures.length, 1); assert.equal(t.width, 8); assert.equal(t.failed, false);
  t.dispose(); gone(f);
});

test('zero extent suspends and restores a fresh mip pyramid without extra services', async () => {
  const f = gpuTargetMipmapFixture(), t = make(f, {width: 0});
  assert.equal(t.mipLevelCount, 0); assert.equal(t.withTexture(() => assert.fail()), false);
  t.resize(8, 4); t.withTexture(() => {}); t.resize(0, 0); await t.whenIdle();
  assert.equal(t.allocatedBytes, 0); assert.equal(t.mipLevelCount, 0);
  t.resize(1, 1); t.withTexture(() => {}); assert.equal(t.mipLevelCount, 1);
  t.dispose(); gone(f);
});

for (const stage of ['pipeline', 'submit', 'validation', 'bindGroup', 'view']) {
  test(`mip ${stage} failure is terminal, observed, and releases all owned storage`, async () => {
    const f = gpuTargetMipmapFixture(), t = make(f), error = Error(`injected ${stage}`);
    if (stage === 'validation') f.controls.validation = {message: error.message};
    else if (stage === 'bindGroup') f.device.createBindGroup = () => { throw error; };
    else if (stage === 'view') f.textures[0].createView = () => { throw error; };
    else f.controls[stage] = error;
    if (stage === 'validation') t.withTexture(() => {});
    else assert.throws(() => t.withTexture(() => assert.fail('failed generation must not publish')), e => e === error);
    await assert.rejects(t.whenIdle(), new RegExp(`injected ${stage}`));
    assert.equal(t.failed, true); assert.equal(t.allocatedBytes, 0); assert.deepEqual(f.scopes, []);
    t.dispose(); gone(f);
  });
}

test('device loss during a stalled mip submission invalidates the target and releases it once', async () => {
  const f = gpuTargetMipmapFixture(), t = make(f);
  t.withTexture(() => {}); f.controls.completion = deferred(); const idle = t.whenIdle();
  f.lost.resolve({message: 'device removed'});
  await assert.rejects(idle, {code: 'GPU_TARGET_DEVICE_LOST'}); gone(f);
  assert.throws(() => t.withTexture(() => assert.fail()), {code: 'GPU_TARGET_DEVICE_LOST'});
  t.dispose(); gone(f);
});
