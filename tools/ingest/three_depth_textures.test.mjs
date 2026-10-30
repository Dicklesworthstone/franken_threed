/** Source DepthTexture storage/lifetime checks. The recorder does not execute
 * depth samples; TRAA browser differentials verify the resulting pixel values. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeDepthTextures} from './three_depth_textures.mjs';
import {createThreeComputeTextures} from './three_compute_textures.mjs';
import {gpuTargetMipmapFixture, deferred} from './fixtures/animation/gpu_target_mipmap_fixture.mjs';

const sampled = texture => ({texture, isSampledTexture: true, isSampler: true});
const sampler = (texture, comparison = false) => ({texture, isSampler: true, comparison});
function fixture(t, options = {}) {
  const f = gpuTargetMipmapFixture();
  const owner = createThreeDepthTextures({three: T, device: f.device, ...options});
  t.after(() => owner.dispose());
  const use = (tokens, callback = () => {}) => owner.withDispatch(tokens, () => {
    const values = tokens.map(token => owner.resolve(token)); callback(values);
    f.device.queue.submit([]); return values;
  });
  return {...f, owner, use};
}

test('detached depth version zero has allocation-free depth metadata without source mutation', t => {
  const f = fixture(t), texture = new T.DepthTexture(1, 1), source = texture.source, image = texture.image;
  const nextId = new T.Texture().id;
  assert.equal(texture.version, 0);
  assert.deepEqual(f.owner.layoutOf(sampled(texture)), {texture: {sampleType: 'depth', viewDimension: '2d', multisampled: false}});
  assert.deepEqual(f.owner.layoutOf(sampler(texture)), {sampler: {type: 'non-filtering'}});
  assert.deepEqual(f.owner.sampleData(texture), {samples: 1, primarySamples: 1, isMSAA: false});
  assert.deepEqual(f.owner.diagnostics, {textures: 0, generations: 0, bindings: 0, bytes: 0});
  assert.equal(f.textures.length, 0); assert.equal(texture.source, source); assert.equal(texture.image, image);
  assert.equal(texture.renderTarget, null); assert.equal(new T.Texture().id, nextId + 1);
});

for (const [type, format] of [[T.UnsignedIntType, 'depth24plus'], [T.FloatType, 'depth32float']])
  test(`detached ${format} uses native initialized storage and source ordinary nearest sampling`, async t => {
    const f = fixture(t), texture = new T.DepthTexture(3, 2, type);
    texture.minFilter = texture.magFilter = T.LinearFilter;
    const read = f.owner.captureBinding(sampled(texture)), sample = f.owner.captureBinding(sampler(texture));
    const [view, nativeSampler] = f.use([read, sample]);
    assert.equal(view.texture.format, format); assert.equal(view.texture.width, 3); assert.equal(view.texture.height, 2);
    assert.equal(view.texture.sampleCount, 1); assert.equal(view.texture.usage, 1 | 2 | 4 | 16);
    assert.equal(view.descriptor.aspect, 'depth-only'); assert.equal(view.descriptor.mipLevelCount, 1);
    assert.equal(nativeSampler.descriptor.minFilter, 'nearest'); assert.equal(nativeSampler.descriptor.magFilter, 'nearest');
    assert.equal(f.owner.diagnostics.bytes, 24); assert.equal(f.textures.length, 1);
    assert.equal(texture.version, 0); assert.equal(texture.isRenderTargetTexture, false);
    f.owner.release(read); f.owner.release(sample); await f.owner.whenIdle(); assert.equal(f.scopes.length, 0);
  });

test('uninitialized or zero source extents allocate one backing texel', async t => {
  const f = fixture(t);
  for (const texture of [new T.DepthTexture(), new T.DepthTexture(0, 0)]) {
    const read = f.owner.captureBinding(sampled(texture)), [view] = f.use([read]);
    assert.equal(view.texture.width, 1); assert.equal(view.texture.height, 1); f.owner.release(read);
  }
  await f.owner.whenIdle(); assert.equal(f.owner.diagnostics.bytes, 8);
});

test('comparison and ordinary samplers share depth while preserving captured filters and compare values', async t => {
  const f = fixture(t), texture = new T.DepthTexture(2, 2);
  texture.compareFunction = T.LessEqualCompare; texture.minFilter = texture.magFilter = T.LinearFilter;
  const read = f.owner.captureBinding(sampled(texture));
  const ordinary = f.owner.captureBinding(sampler(texture)), comparison = f.owner.captureBinding(sampler(texture, true));
  texture.compareFunction = T.GreaterCompare; texture.wrapS = T.RepeatWrapping;
  const later = f.owner.captureBinding(sampler(texture, true));
  const [view, a, b, c] = f.use([read, ordinary, comparison, later]);
  assert.equal(view.texture.format, 'depth24plus'); assert.equal(a.descriptor.compare, undefined);
  assert.equal(a.descriptor.minFilter, 'nearest'); assert.equal(b.descriptor.minFilter, 'linear');
  assert.equal(b.descriptor.compare, 'less-equal'); assert.equal(c.descriptor.compare, 'greater');
  assert.equal(b.descriptor.addressModeU, 'clamp-to-edge'); assert.equal(c.descriptor.addressModeU, 'repeat');
  assert.deepEqual(f.owner.layout(comparison), {sampler: {type: 'comparison'}});
  for (const token of [read, ordinary, comparison, later]) f.owner.release(token);
  await f.owner.whenIdle(); assert.equal(f.textures.length, 1);
});

test('source version changes preserve older queued depth generations and unversioned dimensions stay frozen', async t => {
  const f = fixture(t), texture = new T.DepthTexture(2, 2), first = f.owner.captureBinding(sampled(texture));
  texture.image.width = 4; texture.needsUpdate = true;
  const second = f.owner.captureBinding(sampled(texture)); texture.image.width = 8;
  const same = f.owner.captureBinding(sampled(texture));
  const [a, b, c] = f.use([first, second, same]);
  assert.equal(a.texture.width, 2); assert.equal(b.texture.width, 4); assert.equal(c.texture, b.texture);
  assert.notEqual(a.texture, b.texture); assert.equal(f.owner.diagnostics.bytes, 48);
  f.owner.release(first); await f.owner.whenIdle();
  assert.equal(a.texture.destroyed, 1); assert.equal(b.texture.destroyed, 0);
  assert.equal(f.owner.diagnostics.bytes, 32); f.owner.release(second); f.owner.release(same);
});

test('dispose invalidates queued uses and retires submitted depth only after its completion fence', async t => {
  const f = fixture(t), texture = new T.DepthTexture(1, 1), token = f.owner.captureBinding(sampled(texture));
  f.controls.completion = deferred(); const [view] = f.use([token]); texture.dispose();
  assert.throws(() => f.owner.validateDispatch([token]), {code: 'THREE_DEPTH_TEXTURE_STALE'});
  f.owner.release(token); assert.equal(view.texture.destroyed, 0); assert.equal(f.owner.diagnostics.bytes, 4);
  f.controls.completion.resolve(); await f.owner.whenIdle();
  assert.equal(view.texture.destroyed, 1); assert.equal(f.owner.diagnostics.bytes, 0);
  const fresh = f.owner.captureBinding(sampled(texture)); assert.notEqual(f.use([fresh])[0].texture, view.texture); f.owner.release(fresh);
});

test('all queued depth versions count against budgets before native allocation', async t => {
  const f = fixture(t, {maxBytes: 16, maxBindings: 2}), texture = new T.DepthTexture(2, 2);
  const first = f.owner.captureBinding(sampled(texture)); texture.needsUpdate = true;
  assert.throws(() => f.owner.captureBinding(sampled(texture)), {code: 'THREE_DEPTH_TEXTURE_BUDGET'});
  assert.equal(f.textures.length, 0); f.owner.release(first); texture.dispose();
  const next = f.owner.captureBinding(sampled(texture)), same = f.owner.captureBinding(sampler(texture));
  assert.throws(() => f.owner.captureBinding(sampled(texture)), {code: 'THREE_DEPTH_TEXTURE_BUDGET'});
  f.owner.release(next); f.owner.release(same); await f.owner.whenIdle();
});

test('depth validation failures are observed and prevent later publication', async t => {
  const f = fixture(t), texture = new T.DepthTexture(1, 1), token = f.owner.captureBinding(sampled(texture));
  f.controls.validation = {message: 'bad depth allocation'};
  const [view] = f.use([token]);
  await assert.rejects(f.owner.whenIdle(), {code: 'THREE_DEPTH_TEXTURE_GPU'});
  assert.throws(() => f.owner.resolve(token), {code: 'THREE_DEPTH_TEXTURE_GPU'});
  f.owner.release(token); assert.equal(view.texture.destroyed, 1);
});

for (const cause of ['dispose', 'abort', 'loss'])
  test(`${cause} rejects an outstanding depth wait while physical storage retains its submission fence`, {timeout: 1000}, async t => {
    const lifetime = new AbortController(), f = fixture(t, {signal: lifetime.signal});
    const texture = new T.DepthTexture(1, 1), token = f.owner.captureBinding(sampled(texture));
    const fence = deferred(); f.controls.completion = fence;
    const [view] = f.use([token]), idle = f.owner.whenIdle();
    if (cause === 'dispose') f.owner.dispose();
    if (cause === 'abort') lifetime.abort(Error('renderer cancelled'));
    if (cause === 'loss') f.lost.resolve({message: 'device unavailable'});
    await assert.rejects(idle, cause === 'abort' ? /renderer cancelled/
      : {code: cause === 'dispose' ? 'THREE_DEPTH_TEXTURE_DISPOSED' : 'THREE_DEPTH_TEXTURE_LOST'});
    assert.equal(view.texture.destroyed, 0, 'logical cancellation does not destroy submitted resources early');
    f.owner.release(token); fence.resolve(); await fence.promise;
    assert.equal(view.texture.destroyed, 1); assert.equal(f.owner.diagnostics.bytes, 0);
  });

test('detached depth remains read-only and target-owned, layered or malformed images are refused', t => {
  const f = fixture(t), texture = new T.DepthTexture(1, 1);
  assert.throws(() => f.owner.captureBinding({...sampled(texture), store: true}), {code: 'THREE_DEPTH_TEXTURE_BINDING'});
  const target = new T.RenderTarget(1, 1, {depthTexture: new T.DepthTexture()});
  assert.throws(() => f.owner.layoutOf(sampled(target.depthTexture)), {code: 'THREE_DEPTH_TEXTURE_SOURCE'}); target.dispose();
  for (const change of [x => { x.image.depth = 2; }, x => { x.image.width = -1; }, x => { x.image.data = new Uint32Array(1); },
    x => { x.generateMipmaps = true; }, x => { x.type = T.UnsignedShortType; }]) {
    const value = new T.DepthTexture(1, 1); change(value);
    assert.throws(() => f.owner.captureBinding(sampled(value)), error => /^THREE_DEPTH_TEXTURE_(SHAPE|FORMAT)$/.test(error.code));
  }
  assert.equal(f.textures.length, 0);
});

test('shared graphics/compute owner routes detached depth without CPU uploads or storage writes', async t => {
  const f = fixture(t), writes = [], texture = new T.DepthTexture(1, 1);
  const storage = {validateDispatch(tokens) { assert.deepEqual(tokens, []); }, written(tokens) { writes.push(tokens); }, async whenIdle() {}};
  const owner = createThreeComputeTextures({storage, poolFactory() { assert.fail('Detached depth must not enter a CPU uploader'); }, depthFactory: () => f.owner});
  t.after(() => owner.dispose());
  assert.deepEqual(owner.layoutOf(sampled(texture)), {texture: {sampleType: 'depth', viewDimension: '2d', multisampled: false}});
  const token = owner.captureBinding(sampled(texture)); owner.validateDispatch([token]);
  assert.deepEqual(owner.sampleData(texture), {samples: 1, primarySamples: 1, isMSAA: false});
  owner.withDispatch([token], () => { assert.equal(owner.resolve(token).texture.format, 'depth24plus'); f.device.queue.submit([]); });
  owner.written([token]); assert.deepEqual(writes, [[]]);
  owner.release(token); await owner.whenIdle(); assert.equal(owner.diagnostics.depth.bindings, 0);
});

test('framebuffer copy and sampled depth leases share one captured multisample generation', async t => {
  let samples = 4;
  const f = fixture(t, {getSamples: () => samples}), texture = new T.DepthTexture(3, 2);
  const copy = f.owner.captureFramebuffer(texture), read = f.owner.captureBinding(sampled(texture)), commands = [];
  samples = 1;
  assert.deepEqual(f.owner.sampleData(texture), {samples: 4, primarySamples: 4, isMSAA: false});
  assert.equal(f.owner.layout(read).texture.multisampled, true);
  const source = {format: 'depth24plus', width: 3, height: 2, sampleCount: 4};
  f.owner.withDispatch([copy, read], () => {
    const view = f.owner.resolve(read);
    assert.equal(view.texture.sampleCount, 4);
    assert.equal(f.owner.encodeFramebufferCopy(copy, {copyTextureToTexture(...args) { commands.push(args); }}, source), true);
    assert.equal(commands[0][1].texture, view.texture);
    assert.equal(commands[0][0].aspect, undefined); assert.equal(commands[0][1].aspect, undefined);
    assert.deepEqual(commands[0][2], {width: 3, height: 2, depthOrArrayLayers: 1});
    f.device.queue.submit([]);
  });
  texture.needsUpdate = true;
  const next = f.owner.captureFramebuffer(texture), nextRead = f.owner.captureBinding(sampled(texture));
  assert.equal(f.owner.layout(nextRead).texture.multisampled, false);
  for (const token of [copy, read, next, nextRead]) f.owner.release(token);
  await f.owner.whenIdle(); assert.equal(f.owner.diagnostics.bindings, 0);
});

test('depth copies reject partial extents and sample mismatches before encoding', async t => {
  const f = fixture(t), texture = new T.DepthTexture(3, 2);
  for (const rectangle of [{x: 1, y: 0, z: 3, w: 2}, {x: 0, y: 0, z: 2, w: 2}])
    assert.throws(() => f.owner.captureFramebuffer(texture, {rectangle}), {code: 'THREE_DEPTH_TEXTURE_RECTANGLE'});
  const token = f.owner.captureFramebuffer(texture), encoder = {copyTextureToTexture() { assert.fail('Invalid copy was encoded'); }};
  f.owner.withDispatch([token], () => {
    for (const source of [{width: 4, height: 2, sampleCount: 1}, {width: 3, height: 2, sampleCount: 4}])
      assert.throws(() => f.owner.encodeFramebufferCopy(token, encoder, {format: 'depth24plus', ...source}), {code: 'THREE_DEPTH_TEXTURE_COPY'});
  });
  f.owner.release(token); await f.owner.whenIdle();
});

test('depth format mismatches retain source diagnostics and combined depth budgets', async t => {
  const errors = [], f = fixture(t, {three: {...T, error: (...args) => errors.push(args)}, maxBytes: 16});
  const texture = new T.DepthTexture(2, 1, T.UnsignedInt248Type); texture.format = T.DepthStencilFormat;
  const token = f.owner.captureFramebuffer(texture);
  assert.equal(f.owner.diagnostics.bytes, 16);
  assert.throws(() => f.owner.captureFramebuffer(new T.DepthTexture(1, 1)), {code: 'THREE_DEPTH_TEXTURE_BUDGET'});
  f.owner.withDispatch([token], () => {
    assert.equal(f.owner.encodeFramebufferCopy(token, {copyTextureToTexture() { assert.fail('Format mismatch must not copy'); }},
      {format: 'depth24plus', width: 2, height: 1, sampleCount: 1}), false);
  });
  assert.equal(errors.length, 1);
  assert.deepEqual(errors[0].slice(1), ['depth24plus', 'depth24plus-stencil8']);
  f.owner.release(token); await f.owner.whenIdle();
});
