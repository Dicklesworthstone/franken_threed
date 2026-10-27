/** Retained source cube initialization and queued mip-generation boundaries. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {createThreeRenderTargets} from './three_render_targets.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T);
function fixture(t) {
  const device = geometryDevice(); device.features = new Set(); device.textures = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor});
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  device.queue.writeTexture = () => {};
  let configuration;
  const canvas = {width: 16, height: 16, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [16, 16], format: configuration.format, usage: 16}); },
  })};
  const renderer = new Renderer({canvas, device}); t.after(() => renderer.dispose());
  return {device, renderer};
}
function cube(t, options = {}) {
  const target = new T.CubeRenderTarget(8, {type: T.HalfFloatType, depthBuffer: false,
    generateMipmaps: true, minFilter: T.LinearMipmapLinearFilter, ...options});
  t.after(() => target.dispose()); return target;
}
function producer(t, target) {
  const scene = new T.Scene(), camera = new T.CubeCamera(.1, 10, target);
  const material = new T.MeshBasicNodeMaterial({side: T.BackSide});
  material.colorNode = N.uniform(new T.Color(.25, .5, .75)).setName('cubeSourceColor');
  const geometry = new T.BoxGeometry(4, 4, 4); scene.add(new T.Mesh(geometry, material));
  t.after(() => { geometry.dispose(); material.dispose(); }); return {scene, camera, material};
}
const levels = device => device.passes.map(pass => pass.desc.colorAttachments[0].view.baseMipLevel ?? 0);

test('initRenderTarget initializes the source cube pyramid synchronously without changing selectors or drawing', async t => {
  const f = fixture(t), target = cube(t);
  assert.throws(() => f.renderer.initRenderTarget(target), /initRenderTarget\(\).*before the backend is initialized/);
  await f.renderer.init();
  const selected = new T.RenderTarget(4, 4); t.after(() => selected.dispose());
  f.renderer.setRenderTarget(selected);
  assert.equal(f.renderer.initRenderTarget(target), undefined);
  const entry = f.renderer._targets.capture(target);
  assert.equal(entry.storage.diagnostics.mipLevelCount, 4);
  assert.equal(entry.storage.allocatedBytes, 4080);
  assert.equal(entry.storage.diagnostics.mipGenerations, 0);
  assert.equal(f.renderer.getRenderTarget(), selected); assert.equal(f.renderer.getActiveCubeFace(), 0);
  assert.equal(f.device.submissions.length, 0); assert.equal(target.texture.renderTarget, null);
});

for (const preselect of [false, true]) test(`cold CubeCamera preserves source one-level initialization (metadata selection: ${preselect})`, async t => {
  const f = fixture(t), target = cube(t), p = producer(t, target); await f.renderer.init();
  if (preselect) { f.renderer.setRenderTarget(target); f.renderer.setRenderTarget(null); }
  p.camera.update(f.renderer, p.scene);
  const entry = f.renderer._targets.capture(target);
  assert.equal(target.texture.generateMipmaps, true);
  assert.equal(entry.shape.mipLevelCount, 1);
  await f.renderer.waitForGPU();
  assert.equal(entry.storage.diagnostics.mipLevelCount, 1);
  assert.equal(entry.storage.diagnostics.mipGenerations, 0);
  assert.equal(f.device.textures.filter(texture => texture.depthOrArrayLayers === 6).length, 1);
  assert.deepEqual(levels(f.device), Array(6).fill(0));
  assert.equal(f.renderer.getRenderTarget(), null); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('two queued initialized CubeCamera captures generate after their sixth face and retain call-time flags', async t => {
  const f = fixture(t), target = cube(t), p = producer(t, target); await f.renderer.init();
  f.renderer.initRenderTarget(target);
  const entry = f.renderer._targets.capture(target), texture = f.device.textures.find(value => value.depthOrArrayLayers === 6);
  p.camera.update(f.renderer, p.scene);
  p.material.colorNode.value.setRGB(.75, .5, .25); p.camera.update(f.renderer, p.scene);
  target.texture.generateMipmaps = false;
  await f.renderer.waitForGPU();
  assert.equal(f.renderer._targets.capture(target), entry);
  assert.equal(entry.storage.diagnostics.mipGenerations, 2);
  const capture = [0, 1, 2, 3].flatMap(level => Array(6).fill(level));
  assert.deepEqual(levels(f.device), [...capture, ...capture]);
  assert.ok(f.device.passes.every(pass => pass.desc.colorAttachments[0].view.texture === texture));
  assert.equal(f.renderer.info.f3d.offscreenRenders, 12); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('manual cube clears and sampled borrows do not generate from the current mutable source flag', async t => {
  const f = fixture(t), target = cube(t), p = producer(t, target); await f.renderer.init();
  f.renderer.initRenderTarget(target); p.camera.update(f.renderer, p.scene); await f.renderer.waitForGPU();
  const entry = f.renderer._targets.capture(target);
  f.renderer.setRenderTarget(target, 2); f.renderer.clear(); await f.renderer.waitForGPU();
  f.renderer.setRenderTarget(null);
  const input = f.renderer._targets.captureNativeTexture(target.texture);
  f.renderer._targets.withNativeTexture(input, () => {});
  assert.equal(target.texture.generateMipmaps, true);
  assert.equal(entry.storage.diagnostics.mipGenerations, 1);
  assert.deepEqual(levels(f.device).slice(-1), [0]);
});

test('native cube samplers retain explicit LODs with ordinary linear or nearest minification', async t => {
  const f = fixture(t), target = cube(t); await f.renderer.init(); f.renderer.initRenderTarget(target);
  for (const minFilter of [T.NearestFilter, T.LinearFilter]) {
    target.texture.minFilter = minFilter; target.texture.anisotropy = 8;
    const input = f.renderer._targets.captureNativeTexture(target.texture);
    assert.equal(input.sampler.lodMaxClamp, 3);
    assert.equal(input.sampler.mipmapFilter, 'nearest'); assert.equal(input.sampler.maxAnisotropy, 1);
  }
});

test('authored cube levels remain intact and a new texture version selects a fresh allocation policy', async t => {
  const f = fixture(t), target = cube(t, {generateMipmaps: false}), p = producer(t, target);
  target.texture.mipmaps = [{}, {}, {}]; await f.renderer.init();
  p.camera.activeMipmapLevel = 2; p.camera.update(f.renderer, p.scene); await f.renderer.waitForGPU();
  const old = f.renderer._targets.capture(target);
  assert.equal(old.storage.diagnostics.mipLevelCount, 4); assert.equal(old.storage.diagnostics.mipGenerations, 0);
  assert.deepEqual(levels(f.device), Array(6).fill(2));
  target.texture.mipmaps = []; target.texture.generateMipmaps = true;
  assert.equal(f.renderer._targets.capture(target), old, 'unchanged texture version retains the source allocation');
  target.texture.generateMipmaps = false;
  p.camera.update(f.renderer, p.scene); await f.renderer.waitForGPU();
  assert.deepEqual(levels(f.device).slice(-6), Array(6).fill(2), 'render extents use the allocated levels after descriptor edits');
  target.texture.generateMipmaps = false; target.texture.needsUpdate = true;
  assert.throws(() => f.renderer._targets.assertCurrent(old), {code: 'THREE_TARGET_STALE'});
  f.renderer.initRenderTarget(target);
  const next = f.renderer._targets.capture(target); assert.notEqual(next, old);
  assert.equal(next.storage.diagnostics.mipLevelCount, 1);
});

test('initialized cube mip storage is fully budgeted and public disposal invalidates accepted use', async t => {
  const f = fixture(t), target = cube(t);
  const owner = createThreeRenderTargets(T, {getDevice: () => f.device, maxBytes: 4079}); t.after(() => owner.dispose());
  const entry = owner.capture(target, {initialize: true});
  assert.throws(() => owner.ensure(entry), {code: 'THREE_TARGET_BUDGET'});
  assert.equal(f.device.textures.length, 0);
  await f.renderer.init(); f.renderer.initRenderTarget(target);
  const captured = f.renderer._targets.capture(target), release = f.renderer._targets.retainGeneration(captured);
  target.dispose(); assert.throws(() => f.renderer._targets.assertCurrent(captured), {code: 'THREE_TARGET_STALE'}); release();
});

for (const initialize of [false, true]) test(`unchanged equirectangular conversion survives temporary source disposal (initialized: ${initialize})`, async t => {
  const f = fixture(t), target = cube(t); await f.renderer.init();
  if (initialize) f.renderer.initRenderTarget(target);
  const input = new T.DataTexture(new Uint16Array(16 * 8 * 4).fill(T.DataUtils.toHalfFloat(.5)), 16, 8, T.RGBAFormat, T.HalfFloatType);
  input.needsUpdate = true; t.after(() => input.dispose());
  const generate = input.generateMipmaps, filter = input.minFilter;
  assert.equal(target.fromEquirectangularTexture(f.renderer, input), target);
  await f.renderer.waitForGPU();
  const entry = f.renderer._targets.capture(target);
  assert.equal(entry.storage.diagnostics.mipLevelCount, initialize ? 4 : 1);
  assert.equal(entry.storage.diagnostics.mipGenerations, initialize ? 1 : 0);
  assert.equal(input.generateMipmaps, generate); assert.equal(input.minFilter, filter);
  assert.equal(f.renderer.info.f3d.offscreenRenders, 6); assert.equal(f.renderer._nodeCaptureBytes, 0);
});
