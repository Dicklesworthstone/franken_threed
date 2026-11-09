import assert from 'node:assert/strict';
import test from 'node:test';
import {createThreeRenderTargets, inspectThreeRenderTarget} from './three_render_targets.mjs';
import {targetThree} from './fixtures/target_residency_fixture.mjs';
import {gpuTargetMipmapFixture} from './fixtures/animation/gpu_target_mipmap_fixture.mjs';
import {depthDevice} from './tests/depth_transfer_test_fixture.mjs';

// Minimal source boundary, with r186 DepthTexture defaults and RenderTarget's
// actual association rule. DepthTexture deliberately lacks the color-only flag.
// This is not a replacement for a pinned-Three renderer/pixel integration test.
class DepthTexture extends targetThree.Texture {
  constructor(width = 1, height = 1, type = 1014) {
    super();
    Object.assign(this, {isDepthTexture: true, isRenderTargetTexture: false, renderTarget: null,
      image: {width, height, depth: 1}, type, format: 1026, minFilter: 1003, magFilter: 1003,
      flipY: false, generateMipmaps: false, mipmaps: [], compareFunction: null});
  }
  set needsUpdate(value) { if (value === true) { this.version++; this.source.version++; } }
}
class RenderTarget extends targetThree.RenderTarget {
  // The base boundary constructor calls this setter before other fields exist.
  get depthTexture() { return this._depthTexture ?? null; }
  set depthTexture(current) {
    if (this._depthTexture && this._depthTexture.renderTarget === this) this._depthTexture.renderTarget = null;
    if (current !== null && current.renderTarget === null) current.renderTarget = this;
    this._depthTexture = current;
  }
}
const T = {...targetThree, RenderTarget, DepthTexture, UnsignedIntType: 1014, FloatType: 1015,
  UnsignedShortType: 1012, UnsignedInt248Type: 1020, DepthFormat: 1026, DepthStencilFormat: 1027,
  NeverCompare: 512, LessCompare: 513, EqualCompare: 514, LessEqualCompare: 515,
  GreaterCompare: 516, NotEqualCompare: 517, GreaterEqualCompare: 518, AlwaysCompare: 519,
  LinearMipmapLinearFilter: 1008, NearestMipmapNearestFilter: 1004};
function fixture(options = {}, source = {}) {
  const f = gpuTargetMipmapFixture();
  f.device.features = new Set();
  const owner = createThreeRenderTargets(T, {getDevice: () => f.device, ...options});
  const target = new T.RenderTarget(7, 5, {depthTexture: new T.DepthTexture(), ...source});
  return {...f, owner, target, depth: target.depthTexture};
}
const gone = f => { for (const t of f.textures) assert.equal(t.destroyed, 1); };
const passes = f => f.submissions.flatMap(commands => commands.flatMap(c => c.passes));

for (const [type, format] of [[T.UnsignedIntType, 'depth24plus'], [T.FloatType, 'depth32float']])
  test(`unattached ${format} history remains allocated and sampled while color clears leave it clean`, async () => {
    const f = fixture({}, {depthBuffer: false}), {owner, target, depth} = f;
    depth.type = type;
    const entry = owner.capture(target);
    assert.equal(entry.shape.depthAttachment, false); assert.equal(entry.shape.depthFormat, format);
    assert.equal(entry.shape.bytes, 35 * 8); assert.equal(f.textures.length, 0);
    const binding = owner.bindingsFor().get(depth), native = passes(f)[0].group.entries[0].resource.texture;
    assert.equal(native.format, format); assert.equal(entry.depthDirty, false);
    assert.equal(entry.depthCopies, 1); assert.equal(owner.allocatedBytes, 35 * 12);
    owner.render(entry, frame => {
      assert.equal(Object.hasOwn(frame, 'depthView'), false, 'history storage is not a color-pass attachment');
      const encoder = f.device.createCommandEncoder();
      const pass = encoder.beginRenderPass({colorAttachments: [{view: frame.colorView, loadOp: 'clear',
        storeOp: 'store', clearValue: {r: 1, g: 0, b: 0, a: 1}}]});
      pass.end(); f.device.queue.submit([encoder.finish()]);
    });
    assert.equal(passes(f).at(-1).descriptor.depthStencilAttachment, undefined);
    assert.equal(entry.depthDirty, false);
    assert.equal(owner.bindingsFor().get(depth).view, binding.view); assert.equal(entry.depthCopies, 1);
    let sampled;
    entry.storage.withDepthTexture(texture => { sampled = texture; });
    assert.equal(sampled, native, 'color rendering retains the same depth-history storage');
    await owner.whenIdle(); owner.dispose(); gone(f);
  });

test('copying into unattached history refreshes the comparison replica and survives subsequent color writes', async () => {
  const f = depthDevice(), owner = createThreeRenderTargets(T, {getDevice: () => f.device});
  const source = new T.RenderTarget(4, 3, {depthTexture: new T.DepthTexture(4, 3, T.FloatType)});
  const history = new T.RenderTarget(4, 3, {depthBuffer: false, depthTexture: new T.DepthTexture(4, 3, T.FloatType)});
  const a = owner.capture(source), b = owner.capture(history), values = Float32Array.from({length: 12}, (_, index) => (index + 1) / 16);
  owner.render(a, frame => frame.depthView.texture.data.set(values));
  const bindings = owner.bindingsFor(), color = bindings.get(history.texture), depth = bindings.get(history.depthTexture);
  history.depthTexture.compareFunction = T.LessEqualCompare; history.depthTexture.needsUpdate = true;
  const comparison = bindings.get(history.depthTexture);
  const colorCopies = b.copies, depthCopies = b.depthCopies;
  for (const binding of [depth, comparison]) assert.ok(binding.view.texture.data.every(value => value === 0));
  owner.copyDepth(a, b);
  assert.equal(b.dirty, false); assert.equal(b.depthDirty, true);
  assert.equal(bindings.get(history.texture).view, color.view); assert.equal(b.copies, colorCopies);
  const flipped = [2, 1, 0].flatMap(row => Array.from(values.subarray(row * 4, row * 4 + 4)));
  assert.equal(bindings.get(history.depthTexture).view, comparison.view);
  assert.deepEqual(Array.from(comparison.view.texture.data), flipped);
  assert.equal(b.depthCopies, depthCopies); assert.ok(depth.view.texture.data.every(value => value === 0));
  const submissions = f.submissions.length;
  owner.render(b, frame => {
    assert.equal(Object.hasOwn(frame, 'depthView'), false);
    frame.colorView.texture.data.fill(.875);
  });
  assert.equal(b.depthDirty, true, 'comparison reads never acknowledge the ordinary replica');
  bindings.get(history.depthTexture); assert.equal(b.depthCopies, depthCopies);
  assert.equal(f.submissions.length, submissions, 'a color-only write leaves comparison depth clean');
  b.storage.withDepthTexture(texture => assert.deepEqual(texture.data, values));
  assert.ok(depth.view.texture.data.every(value => value === 0)); assert.equal(depth.view.texture.destroyed, false);
  assert.deepEqual(Array.from(comparison.view.texture.data), flipped);
  await owner.whenIdle(); owner.dispose(); assert.ok(f.textures.every(texture => texture.destroyed));
});

test('unattached history retains queued depth storage across a source effect resize', async () => {
  const f = fixture({}, {depthBuffer: false}), {owner, target, depth} = f;
  const first = owner.ensure(owner.capture(target)), release = owner.retainGeneration(first);
  let original;
  first.storage.withDepthTexture(texture => { original = texture; });
  owner.withNativeEffectTargets([target], () => target.setSize(3, 2));
  assert.equal(owner.assertCurrent(first), first);
  const next = owner.ensure(owner.capture(target));
  assert.notEqual(first, next); assert.equal(first.shape.depthAttachment, false); assert.equal(next.shape.depthAttachment, false);
  assert.deepEqual([first.shape.width, first.shape.height, next.shape.width, next.shape.height], [7, 5, 3, 2]);
  const captured = owner.captureNativeTexture(depth);
  owner.withNativeTexture(captured, texture => {
    assert.equal(texture.width, 3); assert.equal(texture.height, 2); assert.notEqual(texture, original);
  });
  owner.render(first, frame => { assert.equal(Object.hasOwn(frame, 'depthView'), false); assert.equal(frame.colorView.texture.width, 7); });
  first.storage.withDepthTexture(texture => assert.equal(texture, original));
  release(); await owner.whenIdle(); assert.equal(original.destroyed, 1);
  assert.equal(next.alive, true); owner.dispose(); gone(f);
});

for (const samples of [0, 1]) for (const [type, format] of [[T.UnsignedIntType, 'depth24plus'], [T.FloatType, 'depth32float']])
  test(`source ${format}, samples=${samples}: samples the rendered depth as full-precision UV-oriented R32F`, async () => {
    const f = fixture({}, {samples}), {owner, target, depth} = f;
    depth.type = type;
    const image = depth.image, e = owner.capture(target), shape = inspectThreeRenderTarget(target, T);
    assert.equal(f.textures.length, 0, 'selection is GPU-free');
    assert.equal(shape.depthFormat, format); assert.equal(shape.depthTexture, depth);
    assert.equal(shape.bytes, 35 * 8); assert.equal(shape.depthSampleBytes, 35 * 4);
    let written;
    owner.render(e, frame => { written = frame.depthView.texture; });
    assert.equal(written.format, format); assert.equal(written.usage, 1 | 4 | 16);
    assert.equal(f.submissions.length, 0, 'no eager depth copy');
    const map = owner.bindingsFor(), binding = map.get(depth);
    assert.equal(map.has(depth), true); assert.equal(depth.isRenderTargetTexture, false);
    assert.equal(depth.renderTarget, target); assert.equal(depth.image, image, 'source depth image identity stays borrowed');
    assert.deepEqual(depth.image, {width: 7, height: 5, depth: 1});
    assert.equal(binding.view.texture.format, 'r32float'); assert.equal(binding.view.texture.width, 7);
    assert.equal(binding.view.texture.height, 5); assert.equal(binding.view.texture.mipLevelCount, 1);
    assert.equal(binding.view.texture.sampleCount, 1); assert.equal(binding.sampleType, 'unfilterable-float');
    assert.equal(binding.viewDimension, '2d'); assert.equal(binding.version, depth.version); assert.equal(binding.sourceVersion, depth.source.version);
    assert.equal(owner.allocatedBytes, 35 * 12); assert.equal(f.textures.length, 3, 'no color replica allocated by a depth read');
    const [pass] = passes(f), input = pass.group.entries[0].resource;
    assert.equal(input.texture, written); assert.equal(input.descriptor.aspect, 'depth-only');
    assert.notEqual(input.texture, binding.view.texture); assert.equal(pass.vertices, 3);
    assert.equal(pass.pipeline.descriptor.fragment.targets[0].format, 'r32float');
    const code = pass.pipeline.descriptor.fragment.module.code;
    assert.match(code, /var source: texture_depth_2d/);
    assert.match(code, /@location\(0\) f32/);
    assert.match(code, /return textureLoad\(source, vec2i\(i32\(p.x\), i32\(size.y\) - 1 - i32\(p.y\)\), 0\)/);
    assert.equal(e.copies, 0); assert.equal(e.depthCopies, 1);
    assert.equal(e.storage.diagnostics.mipGenerations, 0);
    await owner.whenIdle(); assert.deepEqual(f.scopes, []); owner.dispose(); gone(f);
  });

test('repeated reads reuse the image, bind group, sampler and generation until another write', () => {
  const f = fixture(), {owner, target, depth} = f, e = owner.capture(target), map = owner.bindingsFor();
  const a = map.get(depth), groups = f.groups.length;
  for (let i = 0; i < 5; i++) {
    const b = map.get(depth); assert.equal(b.view, a.view); assert.equal(b.sampler, a.sampler);
  }
  assert.equal(f.submissions.length, 1); assert.equal(f.groups.length, groups);
  owner.render(e, () => {}); const b = map.get(depth);
  assert.equal(b.view, a.view); assert.equal(f.groups.length, groups);
  assert.equal(e.depthCopies, 2); assert.equal(f.submissions.length, 2); assert.equal(f.pipelines.length, 1);
  assert.throws(() => owner.render(e, () => { throw Error('after possible submit'); }), /after possible submit/);
  map.get(depth); assert.equal(e.depthCopies, 3); owner.dispose(); gone(f);
});

test('depth and color pyramids refresh independently and share one complete byte budget', () => {
  const f = fixture({}, {generateMipmaps: true, minFilter: T.LinearMipmapLinearFilter}), {owner, target, depth} = f;
  const e = owner.capture(target), map = owner.bindingsFor();
  owner.render(e, () => {}); const d = map.get(depth);
  assert.equal(e.storage.diagnostics.mipGenerations, 0); assert.equal(f.submissions.length, 1);
  assert.equal(owner.allocatedBytes, (35 + 6 + 1) * 4 + 35 * 4 * 2);
  const c = map.get(target.texture);
  assert.equal(c.view.texture.mipLevelCount, 3); assert.equal(d.view.texture.mipLevelCount, 1);
  assert.equal(e.storage.diagnostics.mipGenerations, 1); assert.equal(f.submissions.length, 3);
  assert.equal(owner.allocatedBytes, 2 * (35 + 6 + 1) * 4 + 35 * 4 * 2);
  owner.render(e, () => {}); map.get(target.texture);
  assert.equal(e.depthCopies, 1, 'a color read must not acknowledge or refresh depth');
  assert.equal(e.storage.diagnostics.mipGenerations, 2); assert.equal(f.submissions.length, 5);
  assert.equal(map.get(depth).view, d.view); assert.equal(e.depthCopies, 2); assert.equal(f.submissions.length, 6);
  owner.dispose(); gone(f);
});

for (const [minFilter, magFilter] of [[T.LinearFilter, T.NearestFilter], [T.NearestFilter, T.LinearFilter], [T.LinearFilter, T.LinearFilter]])
  test(`linear depth sampler ${minFilter}/${magFilter} needs an enabled device feature, not a lossy fallback`, () => {
    const f = fixture(), {owner, depth} = f;
    Object.assign(depth, {minFilter, magFilter});
    assert.throws(() => owner.bindingsFor().get(depth), {code: 'THREE_TARGET_SAMPLER'});
    assert.equal(f.textures.length, 0); assert.equal(f.submissions.length, 0);
    f.device.features.add('float32-filterable');
    const b = owner.bindingsFor().get(depth);
    assert.equal(b.sampleType, 'float'); assert.equal(b.view.texture.format, 'r32float');
    assert.equal(b.sampler.descriptor.minFilter, minFilter === T.LinearFilter ? 'linear' : 'nearest');
    assert.equal(b.sampler.descriptor.magFilter, magFilter === T.LinearFilter ? 'linear' : 'nearest');
    assert.equal(b.sampler.descriptor.lodMaxClamp, 0); owner.dispose(); gone(f);
  });

test('non-filtering depth bindings set every WebGPU filter to nearest, including mipmapFilter', () => {
  const f = fixture(), b = f.owner.bindingsFor().get(f.depth), d = b.sampler.descriptor;
  assert.equal(b.sampleType, 'unfilterable-float');
  for (const key of ['minFilter', 'magFilter', 'mipmapFilter']) assert.equal(d[key], 'nearest');
  assert.equal(d.lodMaxClamp, 0); assert.equal(d.maxAnisotropy, 1);
  assert.equal(d.compare, undefined); f.owner.dispose(); gone(f);
});

for (const samples of [0, 1]) for (const [type, sourceFormat, format, nativeDepthBytes] of [
  [T.UnsignedShortType, T.DepthFormat, 'depth16unorm', 2],
  [T.UnsignedIntType, T.DepthFormat, 'depth24plus', 4],
  [T.FloatType, T.DepthFormat, 'depth32float', 4],
  [T.UnsignedInt248Type, T.DepthStencilFormat, 'depth24plus-stencil8', 8],
  [T.FloatType, T.DepthStencilFormat, 'depth32float-stencil8', 8],
])
  test(`comparison ${format}, samples=${samples}: copies rendered depth into a UV-oriented depth attachment`, async () => {
    const f = fixture({}, {samples}), {owner, target, depth} = f;
    if (format === 'depth32float-stencil8') f.device.features.add('depth32float-stencil8');
    Object.assign(depth, {type, format: sourceFormat, compareFunction: T.LessEqualCompare,
      minFilter: T.LinearFilter, magFilter: T.LinearFilter});
    const entry = owner.capture(target);
    let written;
    owner.render(entry, frame => { written = frame.depthView.texture; });
    const binding = owner.bindingsFor().get(depth), [pass] = passes(f);
    assert.equal(written.format, format);
    assert.equal(binding.sampleType, 'depth'); assert.equal(binding.viewDimension, '2d');
    assert.equal(binding.depthReferenceClamp, type !== T.FloatType);
    assert.equal(binding.view.texture.format, 'depth32float'); assert.equal(binding.view.descriptor.aspect, 'depth-only');
    assert.deepEqual([binding.view.texture.width, binding.view.texture.height, binding.view.texture.mipLevelCount,
      binding.view.texture.sampleCount, binding.view.texture.usage], [7, 5, 1, 1, 4 | 16]);
    assert.deepEqual(binding.samplerDescriptor, binding.sampler.descriptor);
    assert.equal(binding.samplerDescriptor.compare, 'less-equal');
    assert.equal(binding.samplerDescriptor.minFilter, 'linear'); assert.equal(binding.samplerDescriptor.magFilter, 'linear');
    assert.equal(binding.samplerDescriptor.mipmapFilter, 'nearest'); assert.equal(binding.samplerDescriptor.lodMaxClamp, 0);
    assert.equal(binding.samplerDescriptor.maxAnisotropy, 1); assert.equal(f.device.features.has('float32-filterable'), false);
    assert.equal(pass.group.entries[0].resource.texture, written);
    assert.equal(pass.group.entries[0].resource.descriptor.aspect, 'depth-only');
    assert.notEqual(binding.view.texture, written); assert.equal(pass.vertices, 3);
    assert.deepEqual(pass.descriptor.colorAttachments, []);
    assert.equal(pass.descriptor.depthStencilAttachment.view, binding.view);
    assert.equal(pass.descriptor.depthStencilAttachment.depthLoadOp, 'clear');
    assert.equal(pass.descriptor.depthStencilAttachment.depthStoreOp, 'store');
    assert.deepEqual(pass.pipeline.descriptor.fragment.targets, []);
    assert.deepEqual(pass.pipeline.descriptor.depthStencil,
      {format: 'depth32float', depthCompare: 'always', depthWriteEnabled: true});
    const code = pass.pipeline.descriptor.fragment.module.code;
    assert.match(code, /var source: texture_depth_2d/); assert.match(code, /@builtin\(frag_depth\) f32/);
    assert.match(code, /return textureLoad\(source, vec2i\(i32\(p.x\), i32\(size.y\) - 1 - i32\(p.y\)\), 0\)/);
    assert.equal(owner.allocatedBytes, 35 * (8 + nativeDepthBytes)); assert.equal(f.textures.length, 3);
    assert.equal(entry.depthSample, null, 'comparison reads do not allocate a color depth replica');
    assert.equal(depth.isRenderTargetTexture, false); assert.equal(depth.renderTarget, target);
    await owner.whenIdle(); assert.deepEqual(f.scopes, []); owner.dispose(); gone(f);
  });

test('acknowledged comparison function edits preserve the depth view and previous sampler without recopying', () => {
  const f = fixture(), {owner, depth} = f, map = owner.bindingsFor();
  const comparisons = [[T.NeverCompare, 'never'], [T.LessCompare, 'less'], [T.EqualCompare, 'equal'],
    [T.LessEqualCompare, 'less-equal'], [T.GreaterCompare, 'greater'], [T.NotEqualCompare, 'not-equal'],
    [T.GreaterEqualCompare, 'greater-equal'], [T.AlwaysCompare, 'always']];
  Object.assign(depth, {compareFunction: T.LessEqualCompare, minFilter: T.LinearFilter, magFilter: T.LinearFilter});
  const first = map.get(depth), groups = f.groups.length;
  Object.assign(depth, {compareFunction: T.GreaterCompare, wrapS: T.RepeatWrapping, minFilter: T.NearestFilter});
  assert.equal(map.get(depth).sampler, first.sampler, 'unversioned source edits remain invisible');
  depth.wrapS = T.ClampToEdgeWrapping; depth.minFilter = T.LinearFilter;
  for (const [value, compare] of comparisons) {
    depth.compareFunction = value; depth.needsUpdate = true;
    const binding = map.get(depth);
    assert.equal(binding.view, first.view); assert.notEqual(binding.sampler, first.sampler);
    assert.equal(binding.samplerDescriptor.compare, compare);
    assert.equal(binding.samplerDescriptor.minFilter, 'linear'); assert.equal(binding.samplerDescriptor.magFilter, 'linear');
    assert.equal(binding.samplerDescriptor.addressModeU, 'clamp-to-edge'); assert.equal(binding.samplerDescriptor.addressModeV, 'clamp-to-edge');
    assert.equal(binding.version, depth.version); assert.equal(binding.sourceVersion, depth.source.version);
  }
  const last = map.get(depth); depth.compareFunction = null; depth.needsUpdate = true;
  assert.equal(map.get(depth).sampler, last.sampler);
  assert.equal(map.get(depth).samplerDescriptor.compare, 'always', 'r186 never resets the initialized GL compare mode to NONE');
  assert.equal(first.sampler.descriptor.compare, 'less-equal'); assert.equal(first.sampler.descriptor.minFilter, 'linear');
  assert.equal(first.samplerDescriptor.compare, 'less-equal'); assert.ok(Object.isFrozen(first.samplerDescriptor));
  assert.equal(f.groups.length, groups); assert.equal(f.submissions.length, 1); assert.equal(f.textures.length, 3);
  owner.dispose(); gone(f);
});

test('comparison depth requires upload acknowledgement after target allocation and preserves its ordinary replica', () => {
  const f = fixture(), {owner, target, depth} = f, map = owner.bindingsFor(), entry = owner.capture(target);
  owner.render(entry, () => {}); depth.compareFunction = T.LessEqualCompare;
  const ordinary = map.get(depth);
  assert.equal(ordinary.sampleType, 'unfilterable-float', 'first reader uses the producer allocation sampler state');
  depth.needsUpdate = true; const comparison = map.get(depth);
  assert.notEqual(ordinary.view, comparison.view); assert.equal(ordinary.view.texture.format, 'r32float');
  assert.equal(comparison.view.texture.format, 'depth32float'); assert.equal(f.submissions.length, 2);
  assert.equal(owner.allocatedBytes, 35 * 16); assert.equal(f.textures.length, 4);
  depth.compareFunction = null;
  assert.equal(map.get(depth).view, comparison.view, 'an unversioned null cannot disable comparison');
  depth.needsUpdate = true;
  assert.equal(map.get(depth).view, comparison.view, 'acknowledged null retains the source GL comparison mode');
  assert.equal(f.submissions.length, 2); assert.equal(ordinary.view.texture.destroyed, 0);
  owner.render(entry, () => {});
  assert.equal(map.get(depth).view, comparison.view);
  assert.equal(f.submissions.length, 3); assert.equal(passes(f).at(-1).descriptor.depthStencilAttachment.view, comparison.view);
  assert.equal(entry.depthCopies, 1); assert.equal(entry.depthDirty, true, 'comparison copy does not refresh ordinary depth');
  const borrower = owner.capture(new T.RenderTarget(7, 5, {depthTexture: depth}));
  owner.render(borrower, () => {});
  assert.equal(map.get(depth).view, comparison.view); assert.equal(f.submissions.length, 4);
  assert.equal(entry.depthCopies, 1); assert.equal(ordinary.view.texture.destroyed, 0);
  assert.equal(ordinary.sampler.descriptor.compare, undefined); assert.equal(ordinary.view.texture.format, 'r32float');
  assert.equal(f.pipelines.length, 2, 'each depth representation retains one copy pipeline');
  owner.dispose(); gone(f);
});

for (const comparison of [false, true])
test(`multisampled ${comparison ? 'comparison' : 'ordinary'} depth cannot silently enter a single-sample replica`, () => {
  const f = fixture({}, {samples: 4}), entry = f.owner.capture(f.target);
  if (comparison) f.depth.compareFunction = T.LessEqualCompare;
  assert.equal(entry.shape.sampleCount, 4);
  assert.throws(() => f.owner.bindingsFor().get(f.depth), {code: 'THREE_TARGET_UNSUPPORTED'});
  assert.equal(f.textures.length, 0); assert.equal(f.submissions.length, 0); assert.equal(entry.depthSample, null);
  f.owner.dispose();
});

test('unversioned filter and wrap edits stay invisible and acknowledged storage-key changes require recreation', () => {
  const f = fixture(), {owner, depth, target} = f, map = owner.bindingsFor(), a = map.get(depth);
  Object.assign(depth, {wrapS: T.RepeatWrapping, wrapT: T.MirroredRepeatWrapping, minFilter: T.LinearFilter});
  const b = map.get(depth);
  assert.equal(a.view, b.view); assert.equal(a.sampler, b.sampler);
  assert.equal(b.sampler.descriptor.addressModeU, 'clamp-to-edge'); assert.equal(b.sampler.descriptor.addressModeV, 'clamp-to-edge');
  assert.equal(b.sampler.descriptor.minFilter, 'nearest'); assert.equal(f.textures.length, 3);
  assert.equal(f.submissions.length, 1); assert.equal(owner.capture(target).depthCopies, 1);
  depth.needsUpdate = true;
  assert.throws(() => map.get(depth), {code: 'THREE_TARGET_SAMPLER'});
  assert.equal(a.view.texture.destroyed, 0); assert.equal(f.submissions.length, 1);
  target.dispose(); f.device.features.add('float32-filterable');
  const recreated = map.get(depth);
  assert.notEqual(recreated.view, a.view); assert.equal(a.view.texture.destroyed, 1);
  assert.equal(recreated.sampler.descriptor.addressModeU, 'repeat'); assert.equal(recreated.sampler.descriptor.addressModeV, 'mirror-repeat');
  assert.equal(recreated.sampler.descriptor.minFilter, 'linear'); assert.equal(f.submissions.length, 2);
  owner.dispose(); gone(f);
});

for (const [name, mutate] of [
  ['unpack alignment', depth => { depth.unpackAlignment = 8; }],
  ['premultiplication', depth => { depth.premultiplyAlpha = true; }],
  ['third wrapping axis', depth => { depth.wrapR = T.RepeatWrapping; }],
  ['source identity', depth => { depth.source = {version: depth.source.version}; }],
]) test(`acknowledged depth ${name} edits cannot silently replace a source image with retained contents`, () => {
  const f = fixture(), {owner, depth} = f;
  depth.compareFunction = T.LessEqualCompare;
  const binding = owner.bindingsFor().get(depth), submissions = f.submissions.length;
  mutate(depth);
  assert.equal(owner.bindingsFor().get(depth).view, binding.view, 'unversioned changes retain the installed source image');
  depth.needsUpdate = true;
  assert.throws(() => owner.bindingsFor().get(depth), {code: 'THREE_TARGET_SAMPLER'});
  assert.equal(binding.view.texture.destroyed, 0); assert.equal(f.submissions.length, submissions);
  owner.dispose(); gone(f);
});

test('same-format replacement of the DepthTexture invalidates queued generations and removes old listeners', () => {
  const f = fixture(), {owner, target, depth} = f, map = owner.bindingsFor(), e = owner.capture(target);
  const a = map.get(depth), old = f.textures.slice();
  const replacement = new T.DepthTexture(); target.depthTexture = replacement;
  assert.equal(depth.renderTarget, null); assert.equal(map.has(depth), false);
  assert.throws(() => owner.assertCurrent(e), {code: 'THREE_TARGET_STALE'});
  const b = map.get(replacement);
  assert.notEqual(a.view, b.view); assert.ok(old.every(t => t.destroyed === 1));
  assert.equal(depth.listeners.get('dispose').size, 0); assert.equal(replacement.listeners.get('dispose').size, 1);
  depth.dispose(); assert.equal(b.view.texture.destroyed, 0);
  assert.equal(owner.allocatedBytes, 35 * 12); owner.dispose(); gone(f);
});

test('adding/removing depth sampling invalidates storage even when depth24plus is unchanged', () => {
  const f = fixture({}, {depthTexture: null}), {owner, target} = f;
  const first = owner.ensure(owner.capture(target)), old = f.textures.slice();
  target.depthTexture = new T.DepthTexture();
  assert.throws(() => owner.assertCurrent(first), {code: 'THREE_TARGET_STALE'});
  const sampled = owner.capture(target); owner.bindingsFor().get(target.depthTexture);
  assert.ok(old.every(t => t.destroyed === 1));
  const sampledTextures = f.textures.slice(old.length), depth = target.depthTexture;
  target.depthTexture = null;
  assert.throws(() => owner.assertCurrent(sampled), {code: 'THREE_TARGET_STALE'});
  owner.ensure(owner.capture(target));
  assert.ok(sampledTextures.every(t => t.destroyed === 1));
  assert.equal(depth.renderTarget, null); assert.equal(depth.listeners.get('dispose').size, 0);
  assert.equal(owner.allocatedBytes, 35 * 8); owner.dispose(); gone(f);
});

test('changing the source depth type replaces native storage and its sampling view', () => {
  const f = fixture(), {owner, depth, target} = f, map = owner.bindingsFor(), e = owner.capture(target);
  const a = map.get(depth); depth.type = T.FloatType;
  assert.throws(() => owner.assertCurrent(e), {code: 'THREE_TARGET_STALE'});
  const b = map.get(depth); assert.notEqual(a.view, b.view);
  assert.equal(passes(f).at(-1).group.entries[0].resource.texture.format, 'depth32float');
  owner.dispose(); gone(f);
});

for (const comparison of [false, true]) for (const action of ['target', 'color', 'depth', 'owner'])
  test(`${action} disposal releases native, color and ${comparison ? 'comparison' : 'ordinary'} depth resources`, () => {
    const f = fixture(), {owner, target, depth} = f, e = owner.capture(target), map = owner.bindingsFor();
    if (comparison) depth.compareFunction = T.LessEqualCompare;
    map.get(depth); map.get(target.texture);
    ({target, color: target.texture, depth, owner})[action].dispose();
    gone(f); assert.equal(owner.allocatedBytes, 0); assert.equal(owner.size, 0);
    for (const source of [target, target.texture, depth]) assert.equal(source.listeners.get('dispose').size, 0);
    assert.throws(() => owner.assertCurrent(e), {code: action === 'owner' ? 'THREE_TARGET_DISPOSED' : 'THREE_TARGET_STALE'});
    owner.dispose(); gone(f);
  });

test('source resize initializes current depth image extents without replacing its identity', () => {
  const f = fixture(), {owner, target, depth} = f, e = owner.capture(target), map = owner.bindingsFor();
  map.get(depth); const old = f.textures.slice(), image = depth.image; target.setSize(3, 9);
  assert.throws(() => owner.ensure(e), {code: 'THREE_TARGET_STALE'});
  const b = map.get(depth);
  assert.equal(b.view.texture.width, 3); assert.equal(b.view.texture.height, 9);
  assert.equal(owner.allocatedBytes, 3 * 9 * 12); assert.ok(old.every(t => t.destroyed === 1));
  assert.equal(depth.image, image); assert.deepEqual(depth.image, {width: 3, height: 9, depth: 1});
  owner.dispose(); gone(f);
});

for (const comparison of [false, true])
test(`held ${comparison ? 'comparison' : 'ordinary'} depth bindings retain stale storage until released after resize or disposal`, async () => {
  const f = fixture(), {owner, target, depth} = f, map = owner.bindingsFor();
  if (comparison) depth.compareFunction = T.LessEqualCompare;
  const first = map.get(depth), releaseFirst = first.retain(), old = f.textures.slice(), oldBytes = owner.allocatedBytes;
  first.validate(); target.setSize(3, 2);
  assert.throws(first.validate, {code: 'THREE_TARGET_STALE'});
  assert.ok(old.every(texture => texture.destroyed === 0));
  const next = map.get(depth), releaseNext = next.retain();
  assert.notEqual(next.view, first.view); assert.deepEqual([next.view.texture.width, next.view.texture.height], [3, 2]);
  assert.equal(owner.allocatedBytes, oldBytes + 3 * 2 * 12);
  releaseFirst(); releaseFirst(); await owner.whenIdle();
  assert.ok(old.every(texture => texture.destroyed === 1)); assert.equal(owner.allocatedBytes, 3 * 2 * 12);
  next.validate(); depth.dispose();
  assert.throws(next.validate, {code: 'THREE_TARGET_STALE'});
  assert.equal(next.view.texture.destroyed, 0, 'accepted program preparation still holds its retired view');
  releaseNext(); releaseNext(); await owner.whenIdle();
  assert.equal(owner.allocatedBytes, 0); gone(f); owner.dispose(); gone(f);
});

for (const comparison of [false, true])
test(`${comparison ? 'comparison' : 'ordinary'} depth feedback is rejected before allocation and independent targets share only the pipeline`, () => {
  const f = fixture(), {owner, target, depth} = f;
  if (comparison) depth.compareFunction = T.LessEqualCompare;
  const own = owner.bindingsFor(target);
  for (const t of [target.texture, depth]) assert.throws(() => own.get(t), {code: 'THREE_TARGET_FEEDBACK'});
  assert.equal(f.textures.length, 0); assert.equal(owner.size, 0);
  const other = new T.RenderTarget(7, 5, {depthTexture: new T.DepthTexture(7, 5, T.FloatType)});
  if (comparison) other.depthTexture.compareFunction = T.LessEqualCompare;
  const a = own.get(other.depthTexture), b = owner.bindingsFor(other).get(depth);
  assert.notEqual(a.view, b.view); assert.equal(f.pipelines.length, 1);
  assert.equal(owner.allocatedBytes, 2 * 35 * 12); owner.dispose(); gone(f);
});

test('depth texture and detached/external texture bindings retain distinct ownership', () => {
  const f = fixture(), {owner, depth} = f;
  const external = new T.DepthTexture(), retained = new Map([[external, {view: {}, sampler: {}}]]);
  const bindings = owner.bindingsFor(null, retained);
  assert.equal(bindings.get(external), retained.get(external)); assert.equal(bindings.has(external), true);
  assert.equal(bindings.has(depth), true); assert.ok(bindings.get(depth).view);
  retained.delete(external); assert.equal(bindings.has(external), false); assert.equal(bindings.get(external), undefined);
  owner.dispose(); gone(f);
});

test('shared-depth ownership retains the original association and native image', () => {
  const f = fixture(), other = new T.RenderTarget(7, 5, {depthTexture: f.depth});
  assert.equal(f.depth.renderTarget, f.target, 'r186 retains the first association');
  const borrower = f.owner.capture(other);
  assert.equal(f.textures.length, 0);
  const source = f.owner.ensure(f.owner.capture(f.target));
  let depth;
  f.owner.render(source, frame => { depth = frame.depthView.texture; });
  f.owner.render(borrower, frame => assert.equal(frame.depthView.texture, depth));
  f.owner.dispose();
});

test('unsupported depth profiles fail before allocating native attachments or mirrors', () => {
  const edits = [t => t.depthTexture.type = 999,
    t => t.depthTexture.format = T.DepthStencilFormat, t => t.depthTexture.compareFunction = 999,
    t => t.depthTexture.generateMipmaps = true, t => t.depthTexture.mipmaps = [{}],
    t => t.depthTexture.colorSpace = T.SRGBColorSpace, t => t.depthTexture.internalFormat = 'DEPTH_COMPONENT32F',
    t => t.depthTexture.flipY = true, t => t.depthTexture.image.depth = 0, t => t.depthTexture.isCubeTexture = true,
    t => t.depthTexture.minFilter = T.LinearMipmapLinearFilter, t => t.depthTexture.magFilter = 999,
    t => t.depthTexture.wrapT = 999, t => t.depthTexture.anisotropy = 2, t => t.depthTexture.anisotropy = NaN];
  for (const edit of edits) {
    const f = fixture(); edit(f.target);
    assert.throws(() => f.owner.capture(f.target), /THREE_TARGET_/);
    assert.equal(f.textures.length, 0); assert.equal(f.submissions.length, 0); f.owner.dispose();
  }
});

test('depth replica budget failure is retryable and preserves native rendered depth', () => {
  const f = fixture({maxBytes: 419}), {owner, target, depth} = f, e = owner.ensure(owner.capture(target));
  assert.equal(owner.allocatedBytes, 280);
  assert.throws(() => owner.bindingsFor().get(depth), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(e.alive, true); assert.equal(f.textures.length, 2); assert.equal(f.submissions.length, 0);
  target.setSize(3, 3); owner.bindingsFor().get(depth);
  assert.equal(owner.allocatedBytes, 108); owner.dispose(); gone(f);
});

test('shared budget charges the depth image alongside an already allocated color image', () => {
  const f = fixture({maxBytes: 559}), {owner, target, depth} = f, e = owner.capture(target), map = owner.bindingsFor();
  map.get(target.texture); assert.equal(owner.allocatedBytes, 420);
  assert.throws(() => map.get(depth), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(e.alive, true); assert.equal(e.depthCopies, 0); assert.equal(f.textures.length, 3);
  owner.dispose(); gone(f);
});

test('comparison depth replica budget failure preserves the ordinary image and retries at a smaller size', () => {
  const f = fixture({maxBytes: 559}), {owner, target, depth} = f, map = owner.bindingsFor();
  const ordinary = map.get(depth), old = f.textures.slice();
  assert.equal(owner.allocatedBytes, 420);
  depth.compareFunction = T.LessEqualCompare; depth.needsUpdate = true;
  assert.throws(() => map.get(depth), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(owner.allocatedBytes, 420); assert.equal(f.textures.length, 3); assert.equal(f.submissions.length, 1);
  assert.equal(ordinary.view.texture.destroyed, 0); ordinary.validate();
  assert.equal(ordinary.sampler.descriptor.compare, undefined);
  target.setSize(3, 3);
  const comparison = map.get(depth);
  assert.equal(comparison.sampleType, 'depth'); assert.equal(comparison.view.texture.format, 'depth32float');
  assert.equal(owner.allocatedBytes, 108); assert.ok(old.every(texture => texture.destroyed === 1));
  owner.dispose(); gone(f);
});

test('depth disposal in a render callback cannot release a borrowed attachment early', () => {
  const f = fixture(), {owner, target, depth} = f, e = owner.capture(target);
  owner.bindingsFor().get(depth);
  assert.throws(() => owner.render(e, () => {
    depth.dispose(); assert.ok(f.textures.every(t => t.destroyed === 0));
  }), {code: 'THREE_TARGET_STALE'});
  gone(f); owner.dispose();
});

test('sampling during a target submission is rejected even through a destination-free binding map', () => {
  const f = fixture(), {owner, target, depth} = f, e = owner.capture(target), map = owner.bindingsFor();
  owner.render(e, () => assert.throws(() => map.get(depth), {code: 'THREE_TARGET_REENTRANT'}));
  assert.equal(f.textures.length, 2); assert.equal(f.submissions.length, 0);
  map.get(depth); assert.equal(e.depthCopies, 1); owner.dispose(); gone(f);
});

for (const comparison of [false, true]) for (const stage of ['texture', 'view', 'pipeline', 'group', 'sampler', 'submit', 'validation'])
  test(`${comparison ? 'comparison' : 'ordinary'} depth ${stage} failure unwinds native borrowing and releases every resource`, async () => {
    const f = fixture(), {owner, target, depth} = f;
    if (comparison) depth.compareFunction = T.LessEqualCompare;
    owner.ensure(owner.capture(target));
    const error = Error(`injected depth ${stage}`);
    if (stage === 'texture') f.device.createTexture = () => { throw error; };
    if (stage === 'view') {
      const create = f.device.createTexture;
      f.device.createTexture = d => { const t = create(d); t.createView = () => { throw error; }; return t; };
    }
    if (stage === 'pipeline') f.controls.pipeline = error;
    if (stage === 'group') f.device.createBindGroup = () => { throw error; };
    if (stage === 'sampler') f.device.createSampler = () => { throw error; };
    if (stage === 'submit') f.controls.submit = error;
    if (stage === 'validation') f.controls.validation = {message: error.message};
    if (stage === 'validation') owner.bindingsFor().get(depth);
    else assert.throws(() => owner.bindingsFor().get(depth), e => e === error);
    await assert.rejects(owner.whenIdle(), new RegExp(`injected depth ${stage}`));
    assert.equal(owner.allocatedBytes, 0); assert.equal(owner.size, 0); assert.deepEqual(f.scopes, []);
    gone(f); owner.dispose(); gone(f);
  });

test('source disposal during the depth-copy submission defers destruction until its native borrow unwinds', async () => {
  const f = fixture(), {owner, depth} = f, submit = f.device.queue.submit;
  f.device.queue.submit = commands => {
    depth.dispose(); assert.ok(f.textures.every(t => t.destroyed === 0)); submit(commands);
  };
  assert.throws(() => owner.bindingsFor().get(depth), {code: 'THREE_TARGET_STALE'});
  assert.equal(owner.allocatedBytes, 0); gone(f);
  await new Promise(resolve => setImmediate(resolve)); owner.dispose(); gone(f);
});

test('device loss never allows stale native depth to be sampled as a successful new binding', async () => {
  const f = fixture(), {owner, depth} = f;
  owner.bindingsFor().get(depth); f.lost.resolve({message: 'lost depth device'}); await Promise.resolve();
  assert.throws(() => owner.bindingsFor().get(depth), {code: 'THREE_TARGET_GPU'});
  await assert.rejects(owner.whenIdle(), /lost depth device/);
  owner.dispose(); gone(f);
});
