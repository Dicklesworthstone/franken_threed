/** Cube mip ownership and submission contracts; these fixtures do not execute WGSL. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {gpuTargetMipmapFixture, deferred} from './fixtures/animation/gpu_target_mipmap_fixture.mjs';

function fixture(features = []) {
  const f = gpuTargetMipmapFixture(), draws = [];
  f.device.features = new Set(features); f.device.limits.maxTextureArrayLayers = 256;
  const create = f.device.createCommandEncoder;
  f.device.createCommandEncoder = (...args) => {
    const encoder = create(...args), begin = encoder.beginRenderPass;
    encoder.beginRenderPass = descriptor => {
      const pass = begin(descriptor), draw = pass.draw;
      pass.draw = (...values) => { draws.push(values); draw(...values); };
      return pass;
    };
    return encoder;
  };
  return {...f, draws};
}
const make = (f, options = {}) => createGpuRenderTarget(f.device, {
  width: 8, height: 8, cube: true, format: 'rgba16float', depthFormat: null,
  generateMipmaps: true, ...options,
});
const gone = f => { for (const texture of f.textures) assert.equal(texture.destroyed, 1); };

for (const format of ['rgba8unorm', 'rgba8unorm-srgb', 'rgba16float']) {
  for (const size of [8, 7]) test(`${size}x${size} ${format}: cube filtering consumes the complete previous level before each six-face destination`, async t => {
    const f = fixture(), target = make(f, {width: size, height: size, format});
    t.after(() => target.dispose());
    const levels = 1 + Math.floor(Math.log2(size)), bytes = format === 'rgba16float' ? 8 : 4;
    let texels = 0;
    for (let level = 0; level < levels; level++) texels += Math.max(1, Math.floor(size / 2 ** level)) ** 2;
    assert.equal(target.allocatedBytes, texels * 6 * bytes);
    assert.equal(f.submissions.length, 0);
    for (let face = 0; face < 6; face++) target.withFrame(frame => {
      assert.equal(frame.colorView.descriptor.baseArrayLayer, face);
    }, {face});
    assert.equal(f.submissions.length, 0, 'writes alone do not force automatic lazy generation');
    target.withTexture(() => assert.equal(f.submissions.length, 1, 'complete pyramid precedes consumer'));
    const passes = f.submissions[0][0].passes;
    assert.equal(passes.length, (levels - 1) * 6);
    assert.equal(f.groups.length, levels - 1, 'six faces share one immutable source cube view');
    for (let i = 0; i < passes.length; i++) {
      const pass = passes[i], level = 1 + Math.floor(i / 6), face = i % 6;
      const source = pass.group.entries[0].resource, output = pass.descriptor.colorAttachments[0].view;
      assert.deepEqual(source.descriptor, {baseMipLevel: level - 1, mipLevelCount: 1,
        dimension: 'cube', baseArrayLayer: 0, arrayLayerCount: 6});
      assert.deepEqual(output.descriptor, {baseMipLevel: level, mipLevelCount: 1,
        dimension: '2d', baseArrayLayer: face, arrayLayerCount: 1});
      assert.equal(source.texture, output.texture);
      assert.equal(pass.pipeline.descriptor.fragment.targets[0].format, format);
      assert.deepEqual(f.draws[i], [3, 1, 0, face], 'source face direction is selected by firstInstance');
    }
    const code = f.pipelines[0].descriptor.fragment.module.code;
    assert.match(code, /texture_cube<f32>/); assert.match(code, /textureSample\(/);
    assert.doesNotMatch(code, /texture_2d<f32>|textureLoad/);
    assert.deepEqual(f.samplers[0].descriptor, {minFilter: 'linear'});
    const groups = f.groups.slice(); target.withTexture(() => {});
    assert.equal(f.submissions.length, 1, 'unchanged cube does not regenerate');
    target.withFrame(() => {}, {face: 4}); target.withTexture(() => {});
    assert.equal(f.submissions.length, 2); assert.equal(target.diagnostics.mipGenerations, 2);
    assert.deepEqual(f.groups, groups); assert.equal(f.pipelines.length, 1);
    await target.whenIdle(); assert.deepEqual(f.scopes, []);
  });
}

test('explicit generation follows the caller boundary without changing disabled automatic policy', async t => {
  const f = fixture(), target = make(f, {generateMipmaps: false, mipLevelCount: 4});
  t.after(() => target.dispose());
  for (let face = 0; face < 6; face++) target.withFrame(() => {}, {face});
  target.withTexture(() => {}); assert.equal(f.submissions.length, 0);
  assert.equal(target.generateMipmaps(), true); assert.equal(f.submissions.length, 1);
  assert.equal(target.generateMipmaps(), false); assert.equal(f.submissions.length, 1);
  target.withFrame(() => {}, {face: 2, mipLevel: 2}); target.withTexture(() => {});
  assert.equal(f.submissions.length, 1, 'authored levels remain untouched until explicit generation');
  assert.equal(target.generateMipmaps(), true); assert.equal(f.submissions.length, 2);
  assert.equal(target.diagnostics.mipGenerations, 2);
  assert.throws(() => target.withTexture(() => target.generateMipmaps()), {code: 'GPU_TARGET_REENTRANT'});
  await target.whenIdle();
});

test('authored cube mips remain usable without any generation services or optional Float32 filtering', t => {
  const f = fixture(); delete f.device.createRenderPipeline; delete f.device.createSampler;
  const target = make(f, {generateMipmaps: false, mipLevelCount: 4, format: 'rgba32float'});
  t.after(() => target.dispose());
  target.withFrame(() => {}, {face: 5, mipLevel: 3}); target.withTexture(() => {});
  assert.equal(f.submissions.length, 0);
  assert.throws(() => target.generateMipmaps(), {code: 'GPU_TARGET_FEATURE'});
  assert.equal(target.failed, false); target.withTexture(() => {});
});

for (const format of ['r32float', 'rg32float', 'rgba32float']) test(`${format}: cube mip filtering preserves precision and requires the enabled feature`, t => {
  const missing = fixture();
  assert.throws(() => make(missing, {format}), {code: 'GPU_TARGET_FEATURE'});
  assert.equal(missing.textures.length, 0);
  const f = fixture(['float32-filterable']), target = make(f, {format}); t.after(() => target.dispose());
  target.withTexture(() => {});
  assert.equal(f.pipelines[0].descriptor.layout, 'auto');
  assert.equal(f.pipelines[0].descriptor.fragment.targets[0].format, format);
  assert.match(f.pipelines[0].descriptor.fragment.module.code, /texture_cube<f32>/);
  assert.equal(f.samplers.length, 1); assert.equal(f.submissions[0][0].passes.length, 18);
});

test('single-level and suspended cube generation is a no-op, including Float32 without filtering', t => {
  const f = fixture(), target = make(f, {width: 1, height: 1, format: 'rgba32float'}); t.after(() => target.dispose());
  assert.equal(target.generateMipmaps(), false); target.withTexture(() => {});
  assert.equal(f.submissions.length, 0);
  assert.throws(() => target.resize(2, 2), {code: 'GPU_TARGET_FEATURE'});
  assert.equal(target.failed, false); assert.equal(f.textures.length, 1);
  target.resize(0, 0); assert.equal(target.generateMipmaps(), false);
});

test('cube mip cost rejects before allocation and retirement fences retain each full pyramid', async t => {
  const f = fixture(); assert.throws(() => make(f, {maxBytes: 4079}), {code: 'GPU_TARGET_BUDGET'});
  assert.equal(f.textures.length, 0);
  const target = make(f, {maxBytes: 5088}); t.after(() => target.dispose());
  target.generateMipmaps(); const previous = f.textures[0], gate = deferred(); f.controls.completion = gate;
  target.resize(4, 4); assert.equal(target.allocatedBytes, 4080 + 1008); assert.equal(previous.destroyed, 0);
  target.generateMipmaps(); assert.equal(f.submissions[1][0].passes.length, 12);
  assert.throws(() => target.resize(8, 8), {code: 'GPU_TARGET_BUDGET'});
  assert.equal(f.textures.length, 2); assert.equal(target.failed, false);
  gate.resolve(); await target.whenIdle(); assert.equal(previous.destroyed, 1); assert.equal(target.allocatedBytes, 1008);
});

for (const stage of ['pipeline', 'submit', 'validation']) test(`explicit cube generation ${stage} failure remains terminal and releases owned storage`, async () => {
  const f = fixture(), target = make(f, {generateMipmaps: false, mipLevelCount: 4}), error = Error(`cube ${stage}`);
  if (stage === 'validation') f.controls.validation = {message: error.message}; else f.controls[stage] = error;
  if (stage === 'validation') assert.equal(target.generateMipmaps(), true);
  else assert.throws(() => target.generateMipmaps(), cause => cause === error);
  await assert.rejects(target.whenIdle(), new RegExp(`cube ${stage}`));
  assert.equal(target.failed, true); assert.equal(target.allocatedBytes, 0); assert.deepEqual(f.scopes, []);
  target.dispose(); gone(f);
});

test('explicit cube generation disposal and device loss retain the existing cancellation contract', async () => {
  const f = fixture(), target = make(f); target.generateMipmaps();
  const gate = deferred(); f.controls.completion = gate;
  const idle = target.whenIdle(); f.lost.resolve({message: 'cube mip device removed'});
  await assert.rejects(idle, {code: 'GPU_TARGET_DEVICE_LOST'}); gone(f);
  target.dispose(); assert.throws(() => target.generateMipmaps(), {code: 'GPU_TARGET_DISPOSED'}); gone(f);
});
