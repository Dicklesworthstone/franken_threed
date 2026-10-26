/** Actual framebuffer/target owners with recorded WebGPU commands. No pixels or
 * browser claims: these checks establish copy, mip and residency ordering. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeFramebufferTextures} from './three_framebuffer_textures.mjs';
import {createThreeComputeTextures} from './three_compute_textures.mjs';
import {createGpuHdrCanvasRenderer} from './gpu_hdr_canvas.mjs';
import {gpuTargetMipmapFixture, deferred} from './fixtures/animation/gpu_target_mipmap_fixture.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const sampled = texture => ({texture, isSampler: true, isSampledTexture: true});
const sampler = texture => ({texture, isSampler: true});
function fixture(t, options = {}) {
  const f = gpuTargetMipmapFixture(), timelines = [];
  const createEncoder = f.device.createCommandEncoder;
  f.device.createCommandEncoder = () => {
    const encoder = createEncoder(), events = [], begin = encoder.beginRenderPass, finish = encoder.finish;
    encoder.copyTextureToTexture = (source, destination, extent) => events.push({copy: true, source, destination, extent});
    encoder.beginRenderPass = descriptor => { events.push({mip: descriptor.colorAttachments[0].view.descriptor.baseMipLevel}); return begin(descriptor); };
    encoder.finish = () => ({...finish(), events}); timelines.push(events); return encoder;
  };
  const owner = createThreeFramebufferTextures({three: THREE, device: f.device, getFormat: () => 'rgba16float', ...options});
  t.after(() => owner.dispose());
  const source = (width = 4, height = 4, format = 'rgba16float') => ({width, height, format, sampleCount: 1});
  const copy = (tokens, lease, sourceTexture = source(), after = () => {}) => owner.withDispatch(tokens, () => {
    const encoder = f.device.createCommandEncoder(); owner.encodeFramebufferCopy(lease, encoder, sourceTexture); after();
    f.device.queue.submit([encoder.finish()]);
  });
  return {...f, owner, source, copy, timelines};
}
function framebuffer(width = 4, height = 4, mips = true) {
  const texture = new THREE.FramebufferTexture(width, height);
  texture.generateMipmaps = mips; texture.minFilter = THREE.LinearMipmapLinearFilter; texture.magFilter = THREE.LinearFilter;
  return texture;
}

test('framebuffer layout is allocation-free and reports resolved single-sample inputs', t => {
  const f = fixture(t), texture = framebuffer();
  assert.deepEqual(f.owner.layoutOf(sampled(texture)), {texture: {sampleType: 'float', viewDimension: '2d', multisampled: false}});
  assert.deepEqual(f.owner.layoutOf(sampler(texture)), {sampler: {type: 'filtering'}});
  assert.deepEqual(f.owner.sampleData(texture), {samples: 1, primarySamples: 1, isMSAA: false});
  assert.equal(f.textures.length, 0); assert.equal(f.owner.diagnostics.generations, 0);
});

test('copy and mip pyramid share one encoder after the temporary mip flag is restored', async t => {
  const f = fixture(t), texture = framebuffer(5, 3), rectangle = {x: 2, y: 1, z: 5, w: 3};
  const lease = f.owner.captureFramebuffer(texture, {rectangle});
  texture.generateMipmaps = false; rectangle.x = 500;
  const read = f.owner.captureBinding(sampled(texture)), sample = f.owner.captureBinding(sampler(texture));
  let view, nativeSampler;
  f.copy([lease, read, sample], lease, f.source(8, 4), () => {
    view = f.owner.resolve(read); nativeSampler = f.owner.resolve(sample);
  });
  const commands = f.submissions[0][0];
  assert.equal(f.submissions.length, 1); assert.equal(f.timelines.length, 1);
  assert.deepEqual(commands.events.map(event => event.copy ? 'copy' : event.mip), ['copy', 1, 2]);
  assert.deepEqual(commands.events[0].source.origin, {x: 2, y: 1, z: 0});
  assert.equal(commands.events[0].source.mipLevel, 0);
  assert.deepEqual(commands.events[0].destination.origin, {x: 0, y: 0, z: 0});
  assert.equal(view.texture, commands.events[0].destination.texture); assert.equal(view.descriptor.mipLevelCount, 3);
  assert.equal(nativeSampler.descriptor.lodMaxClamp, 2); assert.equal(view.texture.format, 'rgba16float');
  assert.equal(view.texture.sampleCount, 1); assert.equal(view.texture.usage & 7, 7);
  for (const token of [lease, read, sample]) f.owner.release(token);
  await f.owner.whenIdle(); assert.equal(f.scopes.length, 0);
});

test('queued source resize keeps older dimensions and views until their uses complete', async t => {
  const f = fixture(t), texture = framebuffer(), first = f.owner.captureFramebuffer(texture);
  const readFirst = f.owner.captureBinding(sampled(texture));
  texture.image.width = 8; texture.image.height = 2; texture.needsUpdate = true;
  const second = f.owner.captureFramebuffer(texture), readSecond = f.owner.captureBinding(sampled(texture));
  texture.image.width = 999;
  let a, b;
  f.copy([first, readFirst], first, f.source(), () => { a = f.owner.resolve(readFirst); });
  f.copy([second, readSecond], second, f.source(8, 2), () => { b = f.owner.resolve(readSecond); });
  assert.notEqual(a.texture, b.texture); assert.equal(a.texture.width, 4); assert.equal(b.texture.width, 8);
  assert.equal(a.texture.destroyed, 0);
  f.owner.release(first); f.owner.release(readFirst); await f.owner.whenIdle();
  assert.equal(a.texture.destroyed, 1); assert.equal(b.texture.destroyed, 0);
  f.owner.release(second); f.owner.release(readSecond);
});

test('source context clones share image metadata without overwriting earlier captured generations', t => {
  const f = fixture(t), original = framebuffer(), other = original.clone();
  assert.equal(original.image, other.image);
  const first = f.owner.captureFramebuffer(original), a = f.owner.captureBinding(sampled(original));
  other.image.width = 8; other.needsUpdate = true;
  const second = f.owner.captureFramebuffer(other), b = f.owner.captureBinding(sampled(other));
  const laterA = f.owner.captureBinding(sampled(original)); let viewA, viewB;
  f.copy([first, a, laterA], first, f.source(), () => {
    viewA = f.owner.resolve(a); assert.equal(f.owner.resolve(laterA), viewA);
  });
  f.copy([second, b], second, f.source(8, 4), () => { viewB = f.owner.resolve(b); });
  assert.equal(viewA.texture.width, 4); assert.equal(viewB.texture.width, 8);
  for (const token of [first, second, a, b, laterA]) f.owner.release(token);
});

test('repeated copies update the same version while sampler snapshots remain independent', t => {
  const f = fixture(t), texture = framebuffer(), first = f.owner.captureFramebuffer(texture);
  const oldSampler = f.owner.captureBinding(sampler(texture));
  texture.minFilter = THREE.NearestFilter; texture.magFilter = THREE.NearestFilter; texture.wrapS = THREE.RepeatWrapping;
  const second = f.owner.captureFramebuffer(texture), newSampler = f.owner.captureBinding(sampler(texture));
  let a, b;
  f.copy([first, oldSampler], first, f.source(), () => { a = f.owner.resolve(oldSampler); });
  f.copy([second, newSampler], second, f.source(), () => { b = f.owner.resolve(newSampler); });
  assert.equal(f.timelines[0][0].destination.texture, f.timelines[1][0].destination.texture);
  assert.equal(a.descriptor.minFilter, 'linear'); assert.equal(a.descriptor.addressModeU, 'clamp-to-edge');
  assert.equal(b.descriptor.minFilter, 'nearest'); assert.equal(b.descriptor.addressModeU, 'repeat');
  for (const token of [first, second, oldSampler, newSampler]) f.owner.release(token);
});

test('retirement waits for submitted work and source disposal invalidates queued uses', async t => {
  const f = fixture(t), texture = framebuffer(), lease = f.owner.captureFramebuffer(texture);
  f.copy([lease], lease); const native = f.timelines[0][0].destination.texture;
  f.controls.completion = deferred(); texture.dispose();
  assert.throws(() => f.owner.validateDispatch([lease]), {code: 'THREE_FRAMEBUFFER_TEXTURE_STALE'});
  assert.equal(native.destroyed, 0); f.owner.release(lease);
  f.controls.completion.resolve(); await f.owner.whenIdle(); assert.equal(native.destroyed, 1);
});

test('budgets include pending generations and reject before allocating a second texture', async t => {
  const f = fixture(t, {maxBytes: 168}), texture = framebuffer(), lease = f.owner.captureFramebuffer(texture);
  f.copy([lease], lease); assert.equal(f.owner.diagnostics.bytes, 168);
  texture.needsUpdate = true;
  assert.throws(() => f.owner.captureFramebuffer(texture), {code: 'THREE_FRAMEBUFFER_TEXTURE_BUDGET'});
  assert.equal(f.textures.length, 1); f.owner.release(lease);
  texture.dispose(); await f.owner.whenIdle(); assert.equal(f.owner.diagnostics.bytes, 0);
  const next = f.owner.captureFramebuffer(texture); f.owner.release(next);
});

test('copy rejects incompatible formats and unresolvable multisample sources before encoding', t => {
  const f = fixture(t), lease = f.owner.captureFramebuffer(framebuffer());
  for (const source of [f.source(4, 4, 'rgba8unorm'), {...f.source(), sampleCount: 4}, f.source(2, 2)])
    f.owner.withDispatch([lease], () => {
      const encoder = f.device.createCommandEncoder();
      assert.throws(() => f.owner.encodeFramebufferCopy(lease, encoder, source), {code: 'THREE_FRAMEBUFFER_TEXTURE_COPY'});
      assert.equal(f.timelines.at(-1).length, 0);
    });
  f.owner.release(lease);
});

test('shared graphics/compute owner routes framebuffer leases without CPU upload or storage writes', async t => {
  const f = fixture(t), texture = framebuffer(), writes = [];
  const storage = {validateDispatch(tokens) { assert.deepEqual(tokens, []); }, written(tokens) { writes.push(tokens); }, async whenIdle() {}};
  const owner = createThreeComputeTextures({storage, poolFactory() { assert.fail('GPU framebuffer pixels must never enter CPU uploads'); }, framebufferFactory: () => f.owner});
  const lease = owner.captureFramebuffer(texture), read = owner.captureBinding(sampled(texture));
  owner.validateDispatch([lease, read]); let view;
  owner.withDispatch([lease, read], () => {
    const encoder = f.device.createCommandEncoder(); owner.encodeFramebufferCopy(lease, encoder, f.source());
    view = owner.resolve(read); f.device.queue.submit([encoder.finish()]);
  });
  assert.equal(view.texture, f.timelines[0][0].destination.texture);
  owner.written([lease, read]); assert.deepEqual(writes, [[]]);
  owner.release(lease); owner.release(read); await owner.whenIdle();
  assert.throws(() => owner.validateDispatch([read]), {code: 'THREE_FRAMEBUFFER_TEXTURE_STALE'});
  owner.dispose();
});

for (const format of ['rgba8unorm', 'rgba16float', 'rgba32float']) test(`${format} output forwards its resolved linear source and preserves color/depth/stencil loads`, async t => {
  const device = geometryDevice(), renders = []; device.limits.maxTextureDimension2D = 8192;
  device.createTexture = descriptor => {
    const texture = {...descriptor, dimension: '2d', sampleCount: descriptor.sampleCount ?? 1,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(settings = {}) { return {texture, ...settings}; }, destroy() { texture.destroyed = true; }};
    return texture;
  };
  let configuration;
  const canvas = {width: 4, height: 4, getContext: () => ({configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [4, 4], format: configuration.format, usage: configuration.usage}); }})};
  const samples = format === 'rgba32float' ? 1 : 4;
  const session = await createGpuHdrCanvasRenderer(canvas, (_device, attachments) => {
    assert.equal(attachments.format, format); assert.equal(attachments.sampleCount, samples);
    return {render(input, frame, source) {
      assert.equal(source.format, format); assert.equal(source.sampleCount, 1);
      assert.equal((frame.resolveTarget ?? frame.colorView).texture, source); assert.equal(frame.colorView.texture.sampleCount, samples);
      const before = source.recordedValue ?? 0;
      source.recordedValue = (frame.loadOp === 'load' ? before : 0) + input;
      renders.push({frame, source, before, after: source.recordedValue});
    }, async whenIdle() {}, dispose() {}};
  }, {device, renderTarget: {format, sampleCount: samples, depthFormat: 'depth24plus-stencil8'}, output: {toneMapping: 'none'}});
  t.after(() => session.dispose());
  session.render(3, {loadOp: 'clear', depthLoadOp: 'clear', stencilLoadOp: 'clear'});
  session.render(5, {loadOp: 'load', depthLoadOp: 'load', stencilLoadOp: 'load'});
  await session.whenIdle();
  assert.equal(renders[0].source, renders[1].source); assert.equal(renders[1].before, 3); assert.equal(renders[1].after, 8);
  assert.equal(renders[0].frame.depthView, renders[1].frame.depthView);
  assert.equal(renders[1].frame.depthLoadOp, 'load'); assert.equal(renders[1].frame.stencilLoadOp, 'load');
  assert.equal(configuration.usage & 1, 1); assert.equal(device.submissions.length, 2);
});
