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

test('authored mip frames preserve their levels and lazily acquire matching private depth', async () => {
  const f = gpuTargetMipmapFixture(), t = make(f, {generateMipmaps: false, mipLevelCount: 4, depthFormat: 'depth24plus'});
  assert.equal(t.mipLevelCount, 4); assert.equal(t.allocatedBytes, 300);
  assert.equal(t.frameAllocationBytes(1), 32);
  let first;
  t.withFrame(frame => { first = frame; }, {mipLevel: 1});
  assert.equal(first.colorView.descriptor.baseMipLevel, 1); assert.equal(first.colorView.descriptor.mipLevelCount, 1);
  assert.deepEqual([first.depthView.texture.width, first.depthView.texture.height], [4, 2]);
  assert.equal(first.depthView.texture.mipLevelCount, 1); assert.equal(t.allocatedBytes, 332);
  assert.equal(t.frameAllocationBytes(1), 0);
  t.withFrame(frame => assert.equal(frame, first), {mipLevel: 1});
  t.withFrame(frame => {
    assert.equal(frame.colorView.descriptor.baseMipLevel, 3);
    assert.deepEqual([frame.depthView.texture.width, frame.depthView.texture.height], [1, 1]);
    assert.notEqual(frame.depthView.texture, first.depthView.texture);
  }, {mipLevel: 3});
  assert.equal(t.allocatedBytes, 336);
  t.withTexture(texture => assert.equal(texture.mipLevelCount, 4));
  assert.equal(f.submissions.length, 0); assert.equal(f.pipelines.length, 0);
  assert.equal(t.diagnostics.mipGenerations, 0);
  await t.whenIdle(); assert.deepEqual(f.scopes, []); t.dispose(); gone(f);
});

test('private mip depth budget refusal allocates nothing and leaves the target retryable', () => {
  const f = gpuTargetMipmapFixture(), t = make(f, {generateMipmaps: false, mipLevelCount: 4, depthFormat: 'depth24plus', maxBytes: 331});
  const count = f.textures.length;
  assert.throws(() => t.withFrame(() => assert.fail('budget refusal must precede callback'), {mipLevel: 1}), {code: 'GPU_TARGET_BUDGET'});
  assert.equal(t.failed, false); assert.equal(t.allocatedBytes, 300); assert.equal(f.textures.length, count);
  t.resize(0, 0); t.resize(8, 2);
  t.withFrame(frame => assert.deepEqual([frame.depthView.texture.width, frame.depthView.texture.height], [4, 1]), {mipLevel: 1});
  assert.equal(t.allocatedBytes, 172); t.dispose(); gone(f);
});

test('mip selectors reject absent levels, invalid counts, MSAA and sampled depth before callbacks', () => {
  const f = gpuTargetMipmapFixture();
  for (const count of [0, -1, 1.5, '4'])
    assert.throws(() => make(f, {mipLevelCount: count}), {code: 'GPU_TARGET_OPTIONS'});
  assert.throws(() => make(f, {mipLevelCount: 5}), {code: 'GPU_TARGET_SIZE'});
  assert.equal(f.textures.length, 0);
  for (const options of [
    {generateMipmaps: false},
    {mipLevelCount: 4, sampleCount: 4},
    {mipLevelCount: 4, sampleDepth: true, depthFormat: 'depth24plus'},
  ]) {
    const t = make(f, options), count = f.textures.length;
    assert.throws(() => t.withFrame(() => assert.fail('unsupported selector must precede callback'), {mipLevel: 1}), {code: 'GPU_TARGET_MIP'});
    assert.equal(f.textures.length, count); assert.equal(t.failed, false); t.dispose();
  }
  const t = make(f);
  for (const mipLevel of [-1, .5, 4, '1'])
    assert.throws(() => t.withFrame(() => assert.fail(), {mipLevel}), {code: 'GPU_TARGET_MIP'});
  t.dispose(); gone(f);
});

test('explicit mip rendering retains requested automatic regeneration', () => {
  const f = gpuTargetMipmapFixture(), t = make(f, {mipLevelCount: 4});
  t.withTexture(() => {}); assert.equal(t.diagnostics.mipGenerations, 1);
  t.withFrame(frame => assert.equal(frame.colorView.descriptor.baseMipLevel, 2), {mipLevel: 2});
  t.withTexture(() => {}); assert.equal(t.diagnostics.mipGenerations, 2);
  assert.deepEqual(f.submissions[1][0].passes.map(p => p.descriptor.colorAttachments[0].view.descriptor.baseMipLevel), [1, 2, 3]);
  t.dispose(); gone(f);
});

test('throwing mip callbacks retain their depth allocation through resize retirement', async () => {
  const f = gpuTargetMipmapFixture(), t = make(f, {generateMipmaps: false, mipLevelCount: 4, depthFormat: 'depth24plus'});
  assert.throws(() => t.withFrame(() => { throw Error('caller submitted before failure'); }, {mipLevel: 1}), /caller submitted/);
  assert.equal(t.failed, false); assert.equal(t.allocatedBytes, 332);
  const old = f.textures.slice(), fence = deferred(); f.controls.completion = fence;
  t.resize(8, 2); assert.equal(t.allocatedBytes, 332 + 156);
  for (const texture of old) assert.equal(texture.destroyed, 0);
  fence.resolve(); await t.whenIdle();
  for (const texture of old) assert.equal(texture.destroyed, 1);
  assert.equal(t.allocatedBytes, 156); t.dispose(); gone(f);
});

function growthFixture() {
  const f = gpuTargetMipmapFixture();
  Object.assign(f.device.limits, {maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 32, maxTextureArrayLayers: 256});
  return f;
}

for (const sampleCount of [1, 4]) for (const layerCount of [1, 3]) {
  test(`selected mip growth preserves siblings, depth and samples (${sampleCount} samples, ${layerCount} layers)`, async () => {
    const f = growthFixture(), oldLevels = sampleCount === 1 ? 2 : 1;
    const t = createGpuRenderTarget(f.device, {width: 8, height: 4,
      colorFormats: ['rgba16float', 'rgba8unorm'], colorMipLevelCounts: [oldLevels, oldLevels],
      depthFormat: 'depth24plus', sampleCount, ...(layerCount > 1 ? {arrayLayers: layerCount, depthArray: true} : {})});
    const before = new Map();
    for (let face = 0; face < layerCount; face++) for (let mipLevel = 0; mipLevel < oldLevels; mipLevel++)
      t.withFrame(frame => before.set(`${mipLevel}/${face}`, frame), {mipLevel, face});
    t.withFrame(() => f.device.queue.submit([{marker: 'prior draw and resolve'}]));
    let original;
    t.withAttachments(colors => { original = colors; });
    const initialBytes = t.allocatedBytes, textures = f.textures.length, version = t.version;
    const fullBytes = 43 * 4 * layerCount, oldBytes = (oldLevels === 2 ? 40 : 32) * 4 * layerCount;
    assert.equal(t.mipmapGrowthBytes([1]), fullBytes);
    assert.equal(f.textures.length, textures, 'budget inspection allocates nothing');
    assert.equal(f.submissions.length, 1, 'budget inspection submits nothing');
    const fence = deferred(); f.controls.completion = fence;
    assert.equal(t.growMipmaps([1]), true);
    assert.equal(t.version, version, 'content-preserving growth does not change the logical resize epoch');
    assert.equal(t.allocatedBytes, initialBytes + fullBytes, 'old and full replacement colors are charged until the fence');
    assert.deepEqual(t.diagnostics.colorMipLevelCounts, [oldLevels, 4]);
    assert.equal(f.pipelines.length, 0, 'growth does not generate mip pixels');
    const copied = f.submissions[1][0].copies;
    assert.equal(copied.length, oldLevels);
    const replacement = copied[0].destination.texture;
    for (let level = 0; level < oldLevels; level++) {
      assert.equal(copied[level].source.texture, original[1]);
      assert.equal(copied[level].source.mipLevel, level);
      assert.equal(copied[level].destination.texture, replacement);
      assert.equal(copied[level].destination.mipLevel, level);
      assert.deepEqual(copied[level].size, {width: 8 >> level, height: 4 >> level, depthOrArrayLayers: layerCount});
    }
    assert.equal(replacement.mipLevelCount, 4); assert.notEqual(replacement, original[1]);
    t.withAttachments(colors => { assert.equal(colors[0], original[0]); assert.equal(colors[1], replacement); });
    for (let face = 0; face < layerCount; face++) for (let mipLevel = 0; mipLevel < oldLevels; mipLevel++) {
      t.withFrame(frame => {
        const previous = before.get(`${mipLevel}/${face}`);
        assert.equal(frame.depthView, previous.depthView);
        assert.equal(frame.colorViews[0], previous.colorViews[0]);
        if (sampleCount === 4) {
          assert.equal(frame.colorViews, previous.colorViews, 'persistent MSAA views keep every sample');
          assert.equal(frame.resolveTargets[0], previous.resolveTargets[0]);
          assert.equal(frame.resolveTargets[1].texture, replacement);
        } else assert.equal(frame.colorViews[1].texture, replacement);
        const grown = frame.resolveTargets?.[1] ?? frame.colorViews[1];
        assert.equal(grown.descriptor.baseMipLevel, mipLevel);
        if (layerCount > 1) assert.equal(grown.descriptor.baseArrayLayer, face);
      }, {mipLevel, face});
    }
    assert.equal(t.mipmapGrowthBytes([1]), 0); assert.equal(t.growMipmaps([1]), false);
    assert.equal(f.submissions.length, 2);
    assert.equal(t.generateMipmaps([1]), true);
    const filtered = f.submissions[2][0].passes;
    assert.equal(filtered.length, 3 * layerCount);
    for (const pass of filtered) assert.equal(pass.group.entries[0].resource.texture, replacement);
    assert.equal(original[1].destroyed, 0);
    fence.resolve(); await t.whenIdle();
    assert.equal(original[1].destroyed, 1); assert.equal(original[0].destroyed, 0);
    assert.equal(t.allocatedBytes, initialBytes + fullBytes - oldBytes);
    t.dispose(); gone(f);
  });
}

test('mip growth budget failure leaves storage, content bindings and admission usable', async () => {
  const f = growthFixture(), t = createGpuRenderTarget(f.device, {width: 8, height: 4,
    colorFormats: ['rgba8unorm', 'rgba16float'], depthFormat: null, maxBytes: 384 + 344 - 1});
  let frame;
  t.withFrame(value => { frame = value; });
  assert.equal(t.mipmapGrowthBytes([1]), 344);
  assert.throws(() => t.growMipmaps([1]), {code: 'GPU_TARGET_BUDGET'});
  assert.equal(t.failed, false); assert.equal(t.allocatedBytes, 384);
  assert.equal(f.textures.length, 2); assert.equal(f.submissions.length, 0);
  t.withFrame(value => assert.equal(value, frame));
  t.resize(4, 2); await t.whenIdle();
  assert.equal(t.growMipmaps([1]), true, 'a smaller target can retry after the ordinary resize fence');
  await t.whenIdle(); t.dispose(); gone(f);
});

test('grown mip levels participate in transfers and retain full chains after resize', async () => {
  const f = growthFixture(), t = make(f, {generateMipmaps: false, copyDestination: true});
  assert.equal(t.growMipmaps([0]), true);
  const source = make(f, {width: 1, height: 1, generateMipmaps: false});
  assert.equal(t.copyFrom(source, {destinationMipLevel: 3}), true);
  assert.equal(f.submissions[1][0].copies[0].destination.mipLevel, 3);
  t.resize(4, 2); await t.whenIdle();
  assert.equal(t.mipLevelCount, 3); assert.equal(t.allocatedBytes, 11 * 4);
  assert.equal(t.mipmapGrowthBytes([0]), 0);
  t.dispose(); source.dispose(); gone(f);
});

test('mip growth validates selectors and limits growth to admitted owners without affecting no-ops', () => {
  const f = growthFixture(), t = make(f, {generateMipmaps: false});
  for (const colors of [null, [1], [-1], [0, 0], ['0']])
    assert.throws(() => t.mipmapGrowthBytes(colors), {code: 'GPU_TARGET_ATTACHMENT'});
  assert.equal(t.growMipmaps([]), false);
  t.withFrame(() => {
    assert.equal(t.mipmapGrowthBytes([0]), 172);
    assert.throws(() => t.growMipmaps([0]), {code: 'GPU_TARGET_REENTRANT'});
  });
  t.reserveColorHistory([0]);
  assert.throws(() => t.mipmapGrowthBytes([0]), {code: 'GPU_TARGET_MIP'});
  const cube = make(f, {width: 4, height: 4, cube: true, generateMipmaps: false});
  assert.throws(() => cube.growMipmaps([0]), {code: 'GPU_TARGET_MIP'});
  const complete = make(f, {width: 4, height: 4, cube: true});
  assert.equal(complete.mipmapGrowthBytes([0]), 0); assert.equal(complete.growMipmaps([0]), false);
  const suspended = make(f, {width: 0, generateMipmaps: false});
  assert.equal(suspended.mipmapGrowthBytes([0]), 0); assert.equal(suspended.growMipmaps([0]), false);
  t.dispose(); cube.dispose(); complete.dispose(); suspended.dispose(); gone(f);
});

for (const stage of ['allocation', 'view', 'copy', 'submit', 'validation', 'fence']) {
  test(`mip growth ${stage} failure releases partial and current allocations exactly once`, async () => {
    const f = growthFixture(), t = createGpuRenderTarget(f.device, {width: 8, height: 4,
      colorFormats: ['rgba8unorm', 'rgba16float'], depthFormat: 'depth24plus'});
    await t.validation;
    const error = Error(`growth ${stage}`), create = f.device.createTexture;
    if (stage === 'allocation' || stage === 'view') {
      let calls = 0;
      f.device.createTexture = descriptor => {
        if (stage === 'allocation' && ++calls === 2) throw error;
        const texture = create(descriptor);
        if (stage === 'view') texture.createView = () => { throw error; };
        return texture;
      };
    } else if (stage === 'copy') {
      const encode = f.device.createCommandEncoder;
      f.device.createCommandEncoder = () => ({...encode(), copyTextureToTexture() { throw error; }});
    } else if (stage === 'submit') f.controls.submit = error;
    else if (stage === 'validation') f.controls.validation = {message: error.message};
    else f.device.queue.onSubmittedWorkDone = () => { throw error; };
    if (stage === 'validation') assert.equal(t.growMipmaps([0, 1]), true);
    else assert.throws(() => t.growMipmaps([0, 1]), error);
    await assert.rejects(t.whenIdle(), new RegExp(`growth ${stage}`));
    assert.equal(t.failed, true); assert.equal(t.allocatedBytes, 0); assert.deepEqual(f.scopes, []);
    t.dispose(); gone(f);
  });
}
