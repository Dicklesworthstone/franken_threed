import assert from 'node:assert/strict';
import test from 'node:test';
import {createThreeRenderTargets, inspectThreeRenderTarget} from './three_render_targets.mjs';
import {targetThree} from './fixtures/target_residency_fixture.mjs';
import {gpuTargetMipmapFixture} from './fixtures/animation/gpu_target_mipmap_fixture.mjs';

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
  UnsignedShortType: 1012, DepthFormat: 1026, DepthStencilFormat: 1027,
  LessEqualCompare: 515,
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

for (const samples of [0, 1]) for (const [type, format] of [[T.UnsignedIntType, 'depth24plus'], [T.FloatType, 'depth32float']])
  test(`source ${format}, samples=${samples}: samples the rendered depth as full-precision UV-oriented R32F`, async () => {
    const f = fixture({}, {samples}), {owner, target, depth} = f;
    depth.type = type;
    const image = {...depth.image}, e = owner.capture(target), shape = inspectThreeRenderTarget(target, T);
    assert.equal(f.textures.length, 0, 'selection is GPU-free');
    assert.equal(shape.depthFormat, format); assert.equal(shape.depthTexture, depth);
    assert.equal(shape.bytes, 35 * 8); assert.equal(shape.depthSampleBytes, 35 * 4);
    let written;
    owner.render(e, frame => { written = frame.depthView.texture; });
    assert.equal(written.format, format); assert.equal(written.usage, 4 | 16);
    assert.equal(f.submissions.length, 0, 'no eager depth copy');
    const map = owner.bindingsFor(), binding = map.get(depth);
    assert.equal(map.has(depth), true); assert.equal(depth.isRenderTargetTexture, false);
    assert.equal(depth.renderTarget, target); assert.deepEqual(depth.image, image, 'source metadata stays borrowed');
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

test('comparison depth attachments cannot silently enter the WebGL float replica path', () => {
  const f = fixture(), {owner, target, depth} = f;
  depth.compareFunction = T.LessEqualCompare;
  const entry = owner.capture(target);
  assert.equal(entry.shape.depthFormat, 'depth24plus');
  assert.throws(() => owner.bindingsFor().get(depth), {code: 'THREE_TARGET_UNSUPPORTED'});
  assert.equal(f.textures.length, 0); assert.equal(entry.depthSample, null);
  depth.compareFunction = null;
  assert.equal(owner.bindingsFor().get(depth).view.texture.format, 'r32float');
  owner.dispose(); gone(f);
});

test('multisampled depth stays native and cannot silently enter the WebGL float replica path', () => {
  const f = fixture({}, {samples: 4}), entry = f.owner.capture(f.target);
  assert.equal(entry.shape.sampleCount, 4);
  assert.throws(() => f.owner.bindingsFor().get(f.depth), {code: 'THREE_TARGET_UNSUPPORTED'});
  assert.equal(f.textures.length, 0); assert.equal(f.submissions.length, 0); assert.equal(entry.depthSample, null);
  f.owner.dispose();
});

test('sampler and version edits do not regenerate depth or replace its storage', () => {
  const f = fixture(), {owner, depth, target} = f, map = owner.bindingsFor(), a = map.get(depth);
  depth.wrapS = T.RepeatWrapping; depth.wrapT = T.MirroredRepeatWrapping;
  depth.version++; depth.source.version++;
  const b = map.get(depth);
  assert.equal(a.view, b.view); assert.notEqual(a.sampler, b.sampler);
  assert.equal(b.sampler.descriptor.addressModeU, 'repeat'); assert.equal(b.sampler.descriptor.addressModeV, 'mirror-repeat');
  assert.equal(b.version, 1); assert.equal(b.sourceVersion, 1); assert.equal(f.textures.length, 3);
  assert.equal(f.submissions.length, 1); assert.equal(owner.capture(target).depthCopies, 1);
  depth.minFilter = T.LinearFilter;
  assert.throws(() => map.get(depth), {code: 'THREE_TARGET_SAMPLER'});
  assert.equal(a.view.texture.destroyed, 0);
  depth.minFilter = T.NearestFilter; assert.equal(map.get(depth).view, a.view);
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

for (const action of ['target', 'color', 'depth', 'owner'])
  test(`${action} disposal releases native, color and depth resources and detaches listeners`, () => {
    const f = fixture(), {owner, target, depth} = f, e = owner.capture(target), map = owner.bindingsFor();
    map.get(depth); map.get(target.texture);
    ({target, color: target.texture, depth, owner})[action].dispose();
    gone(f); assert.equal(owner.allocatedBytes, 0); assert.equal(owner.size, 0);
    for (const source of [target, target.texture, depth]) assert.equal(source.listeners.get('dispose').size, 0);
    assert.throws(() => owner.assertCurrent(e), {code: action === 'owner' ? 'THREE_TARGET_DISPOSED' : 'THREE_TARGET_STALE'});
    owner.dispose(); gone(f);
  });

test('source resize uses current target extents, not stale DepthTexture image metadata', () => {
  const f = fixture(), {owner, target, depth} = f, e = owner.capture(target), map = owner.bindingsFor();
  map.get(depth); const old = f.textures.slice(); target.setSize(3, 9);
  assert.throws(() => owner.ensure(e), {code: 'THREE_TARGET_STALE'});
  const b = map.get(depth);
  assert.equal(b.view.texture.width, 3); assert.equal(b.view.texture.height, 9);
  assert.equal(owner.allocatedBytes, 3 * 9 * 12); assert.ok(old.every(t => t.destroyed === 1));
  assert.deepEqual(depth.image, {width: 1, height: 1, depth: 1});
  owner.dispose(); gone(f);
});

test('depth and color feedback are rejected before allocation while independent targets share only the pipeline', () => {
  const f = fixture(), {owner, target, depth} = f;
  const own = owner.bindingsFor(target);
  for (const t of [target.texture, depth]) assert.throws(() => own.get(t), {code: 'THREE_TARGET_FEEDBACK'});
  assert.equal(f.textures.length, 0); assert.equal(owner.size, 0);
  const other = new T.RenderTarget(7, 5, {depthTexture: new T.DepthTexture(7, 5, T.FloatType)});
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

test('shared-depth ownership is refused rather than rendering into an unrelated target generation', () => {
  const f = fixture(), other = new T.RenderTarget(7, 5, {depthTexture: f.depth});
  assert.equal(f.depth.renderTarget, f.target, 'r186 retains the first association');
  assert.throws(() => f.owner.capture(other), {code: 'THREE_TARGET_SOURCE'});
  assert.equal(f.textures.length, 0); f.owner.dispose();
});

test('unsupported depth profiles fail before allocating native attachments or mirrors', () => {
  const edits = [t => t.depthBuffer = false, t => t.useArrayDepthTexture = true,
    t => t.depthTexture.type = T.UnsignedShortType, t => t.depthTexture.type = 999,
    t => t.depthTexture.format = T.DepthStencilFormat, t => t.depthTexture.compareFunction = 999,
    t => t.depthTexture.generateMipmaps = true, t => t.depthTexture.mipmaps = [{}],
    t => t.depthTexture.colorSpace = T.SRGBColorSpace, t => t.depthTexture.internalFormat = 'DEPTH_COMPONENT32F',
    t => t.depthTexture.flipY = true, t => t.depthTexture.image.depth = 2, t => t.depthTexture.isArrayTexture = true,
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

for (const stage of ['texture', 'view', 'pipeline', 'group', 'sampler', 'submit', 'validation'])
  test(`depth ${stage} failure stops publication, unwinds native borrowing, and releases every resource`, async () => {
    const f = fixture(), {owner, target, depth} = f;
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
