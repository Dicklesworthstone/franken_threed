import assert from 'node:assert/strict';
import test from 'node:test';
import {createThreeRenderTargets, inspectThreeRenderTarget} from './three_render_targets.mjs';
import {targetThree} from './fixtures/target_residency_fixture.mjs';
import {gpuTargetMipmapFixture} from './fixtures/animation/gpu_target_mipmap_fixture.mjs';
const T = {...targetThree, NearestMipmapNearestFilter: 1004, NearestMipmapLinearFilter: 1005,
  LinearMipmapNearestFilter: 1007, LinearMipmapLinearFilter: 1008};
function fixture(options = {}, source = {}) {
  const f = gpuTargetMipmapFixture(), owner = createThreeRenderTargets(T, {getDevice: () => f.device, ...options});
  const target = new T.RenderTarget(7, 5, {depthBuffer: false, generateMipmaps: true,
    minFilter: T.LinearMipmapLinearFilter, ...source});
  return {...f, owner, target};
}
const gone = f => { for (const texture of f.textures) assert.equal(texture.destroyed, 1); };

for (const samples of [0, 4]) for (const [type, colorSpace, format, bpp] of [
  [T.UnsignedByteType, T.NoColorSpace, 'rgba8unorm', 4],
  [T.UnsignedByteType, T.SRGBColorSpace, 'rgba8unorm-srgb', 4],
  [T.HalfFloatType, T.LinearSRGBColorSpace, 'rgba16float', 8],
]) test(`source NPOT pyramid: ${format}, samples=${samples}, orientation and complete accounting`, async () => {
  const f = fixture({}, {samples, type, colorSpace}), {owner, target} = f;
  const e = owner.capture(target), shape = inspectThreeRenderTarget(target, T);
  assert.equal(f.textures.length, 0); assert.equal(shape.mipLevelCount, 3);
  const colorBytes = (35 + 6 + 1) * bpp, nativeBytes = colorBytes + (samples === 4 ? 35 * bpp * 4 : 0);
  assert.equal(shape.colorBytes, colorBytes); assert.equal(shape.bytes, nativeBytes);
  owner.render(e, frame => assert.equal((frame.resolveTarget ?? frame.colorView).descriptor.mipLevelCount, 1));
  assert.equal(owner.allocatedBytes, nativeBytes);
  const a = owner.bindingsFor().get(target.texture);
  assert.equal(owner.allocatedBytes, nativeBytes + colorBytes);
  assert.equal(a.view.texture.mipLevelCount, 3); assert.equal(a.view.descriptor.mipLevelCount, 3);
  assert.equal(a.sampler.descriptor.lodMaxClamp, 2);
  assert.equal(f.submissions.length, 2, 'native reduction, then orientation');
  const [mips, flipped] = f.submissions.map(commands => commands[0].passes);
  assert.equal(mips.length, 2); assert.equal(flipped.length, 3);
  for (let level = 0; level < flipped.length; level++) {
    const pass = flipped[level], input = pass.group.entries[0].resource, output = pass.descriptor.colorAttachments[0].view;
    assert.equal(input.descriptor.baseMipLevel, level); assert.equal(output.descriptor.baseMipLevel, level);
    assert.equal(input.descriptor.mipLevelCount, 1); assert.equal(output.descriptor.mipLevelCount, 1);
    assert.notEqual(input.texture, output.texture); assert.equal(input.texture.sampleCount, 1);
    assert.equal(pass.pipeline.descriptor.fragment.targets[0].format, format);
  }
  const b = owner.bindingsFor().get(target.texture); assert.equal(a.view, b.view); assert.equal(a.sampler, b.sampler);
  assert.equal(f.submissions.length, 2); assert.equal(e.copies, 1);
  owner.render(e, () => {}); assert.equal(owner.bindingsFor().get(target.texture).view, a.view);
  assert.equal(f.submissions.length, 4); assert.equal(e.copies, 2);
  assert.equal(e.storage.diagnostics.mipGenerations, 2);
  await owner.whenIdle(); assert.deepEqual(f.scopes, []); owner.dispose(); gone(f);
});

for (const [filter, min, mip, lod, aniso] of [
  [T.NearestFilter, 'nearest', 'linear', 0, 1], [T.LinearFilter, 'linear', 'linear', 0, 8],
  [T.NearestMipmapNearestFilter, 'nearest', 'nearest', 2, 1],
  [T.NearestMipmapLinearFilter, 'nearest', 'linear', 2, 1],
  [T.LinearMipmapNearestFilter, 'linear', 'nearest', 2, 1],
  [T.LinearMipmapLinearFilter, 'linear', 'linear', 2, 8],
]) test(`source filter ${filter}: independent texel/mip interpolation and valid anisotropy`, () => {
  const f = fixture({}, {minFilter: filter, anisotropy: 8}), {owner, target} = f;
  const s = owner.bindingsFor().get(target.texture).sampler.descriptor;
  assert.equal(s.minFilter, min); assert.equal(s.mipmapFilter, mip); assert.equal(s.lodMaxClamp, lod);
  assert.equal(s.maxAnisotropy, aniso); owner.dispose(); gone(f);
});

test('sampler edits retain the full pyramid and neither regenerate nor invalidate image views', () => {
  const f = fixture(), {owner, target} = f, bindings = owner.bindingsFor();
  const a = bindings.get(target.texture); target.texture.minFilter = T.NearestFilter; target.texture.version++;
  const b = bindings.get(target.texture);
  assert.equal(a.view, b.view); assert.notEqual(a.sampler, b.sampler); assert.equal(b.sampler.descriptor.lodMaxClamp, 0);
  assert.equal(b.version, 1); assert.equal(f.submissions.length, 2);
  target.texture.minFilter = T.NearestMipmapLinearFilter;
  assert.equal(bindings.get(target.texture).sampler.descriptor.lodMaxClamp, 2);
  assert.equal(f.submissions.length, 2); owner.dispose(); gone(f);
});

test('disabling generation clamps mip samplers to level zero and retains the original allocation profile', () => {
  const f = fixture({}, {generateMipmaps: false}), {owner, target} = f;
  const binding = owner.bindingsFor().get(target.texture);
  assert.equal(binding.view.texture.mipLevelCount, 1); assert.equal(binding.sampler.descriptor.lodMaxClamp, 0);
  assert.equal(owner.allocatedBytes, 35 * 4 * 2); assert.equal(f.submissions.length, 1);
  owner.dispose(); gone(f);
});

test('generation-policy changes invalidate queued generations and replace both full chains', () => {
  const f = fixture(), {owner, target} = f, bindings = owner.bindingsFor();
  const entry = owner.capture(target), first = bindings.get(target.texture), old = f.textures.slice();
  target.texture.generateMipmaps = false;
  assert.throws(() => owner.assertCurrent(entry), {code: 'THREE_TARGET_STALE'});
  const next = bindings.get(target.texture);
  assert.notEqual(first.view, next.view); assert.equal(next.view.texture.mipLevelCount, 1);
  for (const t of old) assert.equal(t.destroyed, 1);
  assert.equal(owner.allocatedBytes, 280); owner.dispose(); gone(f);
});

test('resize and source texture disposal release the mip replicas and lazy native resources', () => {
  const f = fixture(), {owner, target} = f;
  owner.bindingsFor().get(target.texture); target.setSize(1, 9);
  const binding = owner.bindingsFor().get(target.texture);
  assert.equal(binding.view.texture.mipLevelCount, 4); assert.equal(owner.allocatedBytes, (9 + 4 + 2 + 1) * 4 * 2);
  target.texture.dispose(); assert.equal(owner.size, 0); assert.equal(owner.allocatedBytes, 0);
  owner.dispose(); gone(f);
});

test('budgets include both pyramids and reject the replica without losing the rendered native target', () => {
  const f = fixture({maxBytes: 335}), {owner, target} = f, e = owner.ensure(owner.capture(target));
  assert.equal(owner.allocatedBytes, 168);
  assert.throws(() => owner.bindingsFor().get(target.texture), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(f.textures.length, 1); assert.equal(e.alive, true); assert.equal(f.submissions.length, 0);
  target.setSize(3, 3); owner.bindingsFor().get(target.texture); assert.equal(owner.allocatedBytes, 80);
  owner.dispose(); gone(f);
});

test('mip feedback is refused before effects; ping-pong inputs remain distinct', () => {
  const f = fixture(), {owner, target} = f;
  assert.throws(() => owner.bindingsFor(target).get(target.texture), {code: 'THREE_TARGET_FEEDBACK'});
  assert.equal(f.textures.length, 0);
  const other = new T.RenderTarget(7, 5, {depthBuffer: false, generateMipmaps: true});
  const a = owner.bindingsFor(other).get(target.texture), b = owner.bindingsFor(target).get(other.texture);
  assert.notEqual(a.view, b.view); assert.equal(owner.allocatedBytes, 672);
  owner.dispose(); gone(f);
});

for (const step of ['mip', 'flip']) test(`${step} submission failure releases both chains and stops publication`, async () => {
  const f = fixture(), {owner, target} = f, submit = f.device.queue.submit;
  let n = 0;
  f.device.queue.submit = commands => {
    if (++n === (step === 'mip' ? 1 : 2)) throw Error(`${step} submission failed`);
    submit(commands);
  };
  assert.throws(() => owner.bindingsFor().get(target.texture), new RegExp(`${step} submission failed`));
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(owner.allocatedBytes, 0); await assert.rejects(owner.whenIdle(), new RegExp(`${step} submission failed`));
  assert.deepEqual(f.scopes, []); owner.dispose(); gone(f);
});
