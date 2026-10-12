/** Production target owners against call recorders, not GPU pixel evidence. */
import assert from 'node:assert/strict';
import test from 'node:test';
import {createGpuRenderTarget} from './gpu_render_target.mjs';
import {createThreeRenderTargets, inspectThreeRenderTarget} from './three_render_targets.mjs';
import {gpuTargetMipmapFixture, deferred} from './fixtures/animation/gpu_target_mipmap_fixture.mjs';
import {targetThree} from './fixtures/target_residency_fixture.mjs';

const formats = ['depth24plus-stencil8', 'depth32float-stencil8'];
const native = (f, options = {}) => createGpuRenderTarget(f.device, {width: 7, height: 5,
  format: 'rgba8unorm', depthFormat: formats[0], ...options});
const gone = f => f.textures.forEach(t => assert.equal(t.destroyed, 1));
function gpu() {
  const f = gpuTargetMipmapFixture();
  f.device.features = new Set(['depth32float-stencil8']);
  return f;
}

for (const depthFormat of formats) for (const sampleCount of [1, 4])
  for (const sampleDepth of [false, true]) for (const generateMipmaps of [false, true]) {
    test(`native ${depthFormat}/${sampleCount}: sampled=${sampleDepth}, mips=${generateMipmaps}`, async () => {
      const f = gpu(), t = native(f, {depthFormat, sampleCount, sampleDepth, generateMipmaps});
      const colorBytes = (generateMipmaps ? 42 : 35) * 4;
      assert.equal(t.allocatedBytes, colorBytes + (sampleCount === 4 ? 35 * 16 : 0) + 35 * 8 * sampleCount);
      assert.deepEqual(t.rendererOptions, {format: 'rgba8unorm', depthFormat, sampleCount});
      let depth;
      t.withFrame(frame => {
        depth = frame.depthView.texture;
        assert.equal(depth.format, depthFormat);
        assert.equal(depth.sampleCount, sampleCount);
        assert.equal(frame.depthView.descriptor.aspect ?? 'all', 'all', 'rendering retains stencil');
        assert.equal(depth.usage, sampleDepth ? 20 : 16);
        assert.equal(depth.mipLevelCount, 1);
        assert.equal(!!frame.resolveTarget, sampleCount === 4);
      });
      if (sampleDepth) t.withDepthTexture(texture => {
        assert.equal(texture, depth);
        assert.equal(texture.createView({aspect: 'depth-only'}).descriptor.aspect, 'depth-only');
      });
      else assert.throws(() => t.withDepthTexture(() => assert.fail()), {code: 'GPU_TARGET_DEPTH'});
      assert.equal(f.submissions.length, 0, 'depth does not generate color mips');
      t.withTexture(() => {});
      assert.equal(f.submissions.length, generateMipmaps ? 1 : 0);
      for (const commands of f.submissions) for (const pass of commands[0].passes) {
        assert.equal(pass.descriptor.depthStencilAttachment, undefined, 'color reduction cannot clear stencil');
        assert.notEqual(pass.group.entries[0].resource.texture, depth);
      }
      await t.whenIdle(); t.dispose(); gone(f);
    });
  }

test('optional depth32/stencil feature is checked before allocations, even when suspended', () => {
  const f = gpu(); f.device.features.clear();
  for (const width of [0, 7]) assert.throws(() => native(f, {width, depthFormat: formats[1]}), {code: 'GPU_TARGET_FEATURE'});
  assert.equal(f.textures.length, 0);
  const t = native(f); t.dispose(); gone(f);
});

test('combined allocation is budgeted before effects; resize retirement includes all aspects', async () => {
  const f = gpu();
  assert.throws(() => native(f, {maxBytes: 419}), {code: 'GPU_TARGET_BUDGET'});
  assert.equal(f.textures.length, 0);
  const t = native(f, {sampleDepth: true, maxBytes: 444}), fence = deferred();
  t.withDepthTexture(() => {}); f.controls.completion = fence;
  const previous = f.textures.slice(); t.resize(2, 1);
  assert.equal(t.allocatedBytes, 444); previous.forEach(p => assert.equal(p.destroyed, 0));
  assert.throws(() => t.resize(1, 1), {code: 'GPU_TARGET_BUDGET'});
  assert.equal(t.failed, false); assert.equal(f.textures.length, 4);
  fence.resolve(); await t.whenIdle(); previous.forEach(p => assert.equal(p.destroyed, 1));
  assert.equal(t.allocatedBytes, 24); t.resize(1, 1); t.dispose(); gone(f);
});

for (const stop of ['dispose', 'loss', 'validation']) test(`combined target ${stop} releases once and ends waits`, async () => {
  const f = gpu(), t = native(f, {sampleDepth: true}), fence = deferred();
  t.withFrame(() => {}); f.controls.completion = fence;
  if (stop === 'validation') f.controls.validation = {message: 'combined attachment invalid'};
  t.resize(3, 2); const idle = t.whenIdle();
  if (stop === 'dispose') t.dispose();
  if (stop === 'loss') f.lost.resolve({message: 'combined attachment lost'});
  await assert.rejects(idle, /GPU_TARGET_/); gone(f);
  fence.resolve(); await Promise.resolve(); t.dispose(); gone(f);
});

test('combined target suspension and synchronous borrowing retain the established lifetime', async () => {
  const f = gpu(), t = native(f, {width: 0, sampleDepth: true});
  assert.equal(t.withDepthTexture(() => assert.fail()), false);
  t.resize(7, 5);
  assert.throws(() => t.withDepthTexture(() => t.resize(2, 2)), {code: 'GPU_TARGET_REENTRANT'});
  assert.throws(() => t.withDepthTexture(() => Promise.resolve()), {code: 'GPU_TARGET_ASYNC_USE'});
  t.resize(0, 0); await t.whenIdle(); assert.equal(t.allocatedBytes, 0); t.dispose(); gone(f);
});

class DepthTexture extends targetThree.Texture {
  constructor(type = 1020, format = 1027) {
    super(); Object.assign(this, {type, format, isDepthTexture: true, image: {width: 7, height: 5, depth: 1},
      renderTarget: null, flipY: false, compareFunction: null, minFilter: 1003, magFilter: 1003, mipmaps: []});
  }
}
const T = {...targetThree, DepthTexture, DepthFormat: 1026, DepthStencilFormat: 1027,
  UnsignedIntType: 1014, FloatType: 1015, UnsignedInt248Type: 1020,
  NearestMipmapNearestFilter: 1004, NearestMipmapLinearFilter: 1005,
  LinearMipmapNearestFilter: 1007, LinearMipmapLinearFilter: 1008};
function source(options = {}, ownerOptions = {}) {
  const f = gpu(), target = new T.RenderTarget(7, 5, {stencilBuffer: true, ...options});
  const owner = createThreeRenderTargets(T, {getDevice: () => f.device, ...ownerOptions});
  return {...f, owner, target};
}
function attach(target, type = T.UnsignedInt248Type, format = T.DepthStencilFormat) {
  const depth = new T.DepthTexture(type, format); depth.renderTarget = target; target.depthTexture = depth; return depth;
}

for (const samples of [0, 1, 4]) test(`source private stencil: ${samples} samples, no precision or resolve substitution`, async () => {
  const f = source({samples, generateMipmaps: true}), {owner, target} = f;
  const entry = owner.capture(target);
  assert.equal(f.textures.length, 0); assert.equal(entry.shape.depthFormat, formats[0]);
  owner.render(entry, frame => {
    assert.equal(frame.depthView.texture.format, formats[0]);
    assert.equal(frame.depthView.texture.sampleCount, samples || 1);
    assert.equal(frame.depthView.descriptor.aspect ?? 'all', 'all');
  });
  assert.equal(owner.allocatedBytes, entry.storage.allocatedBytes);
  const binding = owner.bindingsFor().get(target.texture);
  assert.equal(binding.view.texture.format, 'rgba8unorm');
  assert.equal(owner.allocatedBytes, entry.shape.bytes + entry.shape.colorBytes);
  assert.equal(f.textures.filter(t => t.format === formats[0]).length, 1);
  await owner.whenIdle(); owner.dispose(); gone(f);
});

for (const stencilBuffer of [false, true]) for (const [type, format, expected] of [
  [T.UnsignedInt248Type, T.DepthStencilFormat, 'depth24plus-stencil8'],
  [T.UnsignedIntType, T.DepthFormat, 'depth24plus'], [T.FloatType, T.DepthFormat, 'depth32float'],
]) test(`source texture selects attachment: ${type}/${format}, stencilBuffer=${stencilBuffer}`, async () => {
  const f = source({stencilBuffer}), {owner, target} = f, depth = attach(target, type, format);
  const image = {...depth.image}, entry = owner.capture(target);
  owner.render(entry, frame => assert.equal(frame.depthView.texture.format, expected));
  const d = owner.bindingsFor().get(depth), pass = f.submissions.at(-1)[0].passes[0];
  const input = pass.group.entries[0].resource;
  assert.equal(input.texture.format, expected); assert.equal(input.descriptor.aspect, 'depth-only');
  assert.equal(d.view.texture.format, 'r32float'); assert.equal(d.sampleType, 'unfilterable-float');
  assert.equal(owner.allocatedBytes, 35 * (expected === formats[0] ? 16 : 12));
  assert.deepEqual(depth.image, image); assert.equal(depth.isRenderTargetTexture, undefined);
  const before = f.submissions.length; owner.bindingsFor().get(depth); assert.equal(f.submissions.length, before);
  owner.render(entry, () => {}); owner.bindingsFor().get(depth); assert.equal(entry.depthCopies, 2);
  await owner.whenIdle(); owner.dispose(); gone(f);
});

test('source stencil changes retire dispatcher, native attachment and replicas before reuse', () => {
  const f = source({stencilBuffer: false}), {owner, target} = f;
  const entry = owner.capture(target); owner.bindingsFor().get(target.texture);
  const old = f.textures.slice(); let retired = 0; entry._dispatcher = {dispose() { retired++; }};
  target.stencilBuffer = true;
  assert.throws(() => owner.assertCurrent(entry), {code: 'THREE_TARGET_STALE'});
  const next = owner.capture(target); owner.ensure(next);
  old.forEach(t => assert.equal(t.destroyed, 1)); assert.equal(retired, 1);
  assert.equal(next.storage.rendererOptions.depthFormat, formats[0]);
  target.dispose(); assert.equal(owner.allocatedBytes, 0); owner.dispose(); gone(f);
});

test('source stencil and depth replicas share a retryable, conservative budget', () => {
  const f = source({}, {maxBytes: 559}), {owner, target} = f, depth = attach(target);
  const entry = owner.ensure(owner.capture(target)); assert.equal(owner.allocatedBytes, 420);
  assert.throws(() => owner.bindingsFor().get(depth), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(entry.alive, true); assert.equal(f.textures.length, 2);
  target.setSize(2, 2); owner.bindingsFor().get(depth); owner.bindingsFor().get(target.texture);
  assert.equal(owner.allocatedBytes, 80); owner.dispose(); gone(f);
});

test('source combined feedback is rejected before native allocation', () => {
  const f = source(), {owner, target} = f, depth = attach(target), bindings = owner.bindingsFor(target);
  for (const t of [target.texture, depth]) assert.throws(() => bindings.get(t), {code: 'THREE_TARGET_FEEDBACK'});
  assert.equal(f.textures.length, 0); owner.dispose();
});

for (const mutation of ['wrong-type', 'multisample', 'compare', 'no-depth'])
  test(`source combined ${mutation} remains an explicit allocation-free refusal`, () => {
    const f = source(), {owner, target} = f, depth = attach(target);
    if (mutation === 'wrong-type') depth.type = T.FloatType;
    if (mutation === 'multisample') target.samples = 4;
    if (mutation === 'compare') depth.compareFunction = 515;
    if (mutation === 'no-depth') target.depthBuffer = false;
    assert.throws(() => inspectThreeRenderTarget(target, T), {code: 'THREE_TARGET_UNSUPPORTED'});
    assert.equal(f.textures.length, 0); owner.dispose();
  });

for (const kind of ['target', 'depth', 'color']) test(`source ${kind} disposal releases combined storage and both replicas`, () => {
  const f = source(), {owner, target} = f, depth = attach(target), entry = owner.capture(target);
  owner.bindingsFor().get(depth); owner.bindingsFor().get(target.texture);
  (kind === 'target' ? target : kind === 'depth' ? depth : target.texture).dispose();
  assert.equal(owner.size, 0); assert.equal(owner.allocatedBytes, 0); assert.equal(entry.lifetime.signal.aborted, true);
  for (const t of [target, target.texture, depth]) assert.equal(t.listeners.get('dispose').size, 0);
  owner.dispose(); gone(f);
});

for (const step of ['allocation', 'sampling', 'validation']) test(`source combined ${step} failures cannot publish a usable image`, async () => {
  const f = source(), {owner, target} = f, depth = attach(target);
  if (step === 'allocation') {
    const create = f.device.createTexture;
    f.device.createTexture = d => { if (d.format === formats[0]) throw Error('combined allocation'); return create(d); };
    assert.throws(() => owner.ensure(owner.capture(target)), /combined allocation/);
  } else {
    const entry = owner.ensure(owner.capture(target));
    if (step === 'sampling') f.controls.submit = Error('combined sampling');
    else f.controls.validation = {message: 'combined validation'};
    if (step === 'sampling') assert.throws(() => owner.bindingsFor().get(depth), /combined sampling/);
    else owner.bindingsFor().get(depth);
    await assert.rejects(owner.whenIdle(), /combined (sampling|validation)/);
    assert.equal(entry.alive, false);
  }
  owner.dispose(); gone(f);
});
