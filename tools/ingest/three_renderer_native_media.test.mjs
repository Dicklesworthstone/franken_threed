/** Genuine source VideoTexture/VideoFrameTexture with recording host/GPU media.
 * Tests source WGSL, synchronous capture and owned-frame lifetimes. Host doubles
 * do not establish decoder behavior, color precision or browser pixel parity.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {hasThreeNodeProgram} from './three_node_program.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T);
function platform(t, {frames = true, defer = false} = {}) {
  const keys = ['HTMLVideoElement', 'VideoFrame', 'ImageBitmap', 'createImageBitmap', 'structuredClone'];
  const saved = new Map(keys.map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  const owned = [], bitmaps = [], pending = [];
  class VideoElement {
    constructor(callbacks = true) {
      Object.assign(this, {videoWidth: 4, videoHeight: 2, width: 800, height: 600, readyState: 2, HAVE_CURRENT_DATA: 2,
        pixels: new Uint8Array(32).fill(17), requests: 0, cancellations: 0, currentTime: 0, paused: true, playbackRate: 1});
      if (callbacks) this.requestVideoFrameCallback = callback => { this.callback = callback; return ++this.requests; };
    }
    frame(value) { this.pixels.fill(value); this.currentTime++; this.callback?.(0, {}); }
    cancelVideoFrameCallback() { this.cancellations++; this.callback = null; }
    play() { assert.fail('Renderer must not start playback'); }
    pause() { assert.fail('Renderer must not alter playback'); }
  }
  class VideoFrame {
    constructor(source, init = {}) {
      if (source instanceof VideoElement && source.readyState < 2 || source.closed)
        throw new DOMException('No current frame', 'InvalidStateError');
      if (source.tainted) throw new DOMException('Cross-origin video', 'SecurityError');
      this.pixels = source instanceof Uint8Array ? source.slice() : source.pixels.slice();
      this.width = init.codedWidth ?? source.width; this.height = init.codedHeight ?? source.height;
      if (source instanceof VideoElement) { this.width = source.videoWidth; this.height = source.videoHeight; }
      this.closeCalls = 0; this.closed = false; owned.push(this);
    }
    get displayWidth() { return this.closed ? 0 : this.width; }
    get displayHeight() { return this.closed ? 0 : this.height; }
    clone() { return new VideoFrame(this); }
    close() { this.closeCalls++; this.closed = true; }
  }
  class ImageBitmap {
    constructor(source) { this.pixels = source.pixels.slice(); this.width = source.videoWidth ?? source.width;
      this.height = source.videoHeight ?? source.height; this.closeCalls = 0; bitmaps.push(this); }
    close() { this.closeCalls++; this.width = this.height = 0; }
  }
  Object.assign(globalThis, {HTMLVideoElement: VideoElement, VideoFrame: frames ? VideoFrame : undefined, ImageBitmap,
    structuredClone: image => new ImageBitmap(image),
    createImageBitmap(source, options) {
      assert.deepEqual(options, {premultiplyAlpha: 'none', colorSpaceConversion: 'default'});
      if (source.tainted) return Promise.reject(new DOMException('Cross-origin video', 'SecurityError'));
      const image = new ImageBitmap(source);
      if (!defer) return Promise.resolve(image);
      return new Promise((resolve, reject) => pending.push({image, resolve, reject}));
    }});
  t.after(() => { for (const [key, descriptor] of saved) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
  }});
  return {VideoElement, VideoFrame, owned, bitmaps, pending,
    settle() { for (const item of pending.splice(0)) item.resolve(item.image); }};
}
function fixture(t, hostOptions = {}, Material = T.MeshBasicNodeMaterial) {
  const host = platform(t, hostOptions), device = geometryDevice(), copies = [], textures = [], captures = [];
  device.features = new Set();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = desc => {
    const value = {dimension: '2d', sampleCount: 1, ...desc,
      width: desc.size.width ?? desc.size[0], height: desc.size.height ?? desc.size[1],
      depthOrArrayLayers: desc.size.depthOrArrayLayers ?? desc.size[2] ?? 1,
      createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }};
    textures.push(value); return value;
  };
  device.createSampler = desc => ({...desc}); device.queue.writeTexture = () => {};
  device.queue.copyExternalImageToTexture = (source, destination, size) => {
    assert.notEqual(source.source.closed, true);
    const pixels = source.source.pixels.slice(); destination.texture.pixels = pixels;
    copies.push({source: {...source}, destination: {...destination}, size: [...size], pixels});
  };
  let configuration;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 32], format: configuration.format, usage: 16}); },
  })};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const material = new Material(), object = new T.Mesh(new T.PlaneGeometry(), material);
  object.frustumCulled = false;
  const scene = new T.Scene(); scene.add(object);
  const camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 4;
  const capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function(request) { const frame = capture.call(this, request); if (frame) captures.push(frame); return frame; };
  t.after(() => { renderer.dispose(); material.dispose(); object.geometry.dispose(); });
  return {host, device, renderer, material, object, scene, camera, copies, textures, captures};
}
function video(t, f, {callbacks = true, requested = true} = {}) {
  const image = new f.host.VideoElement(callbacks), texture = new T.VideoTexture(image); texture.colorSpace = T.SRGBColorSpace;
  if (requested) texture.needsUpdate = true;
  t.after(() => texture.dispose()); return texture;
}
function frameTexture(t, f, value = 17) {
  const frame = new f.host.VideoFrame(new Uint8Array(32).fill(value), {codedWidth: 4, codedHeight: 2, timestamp: 0});
  const texture = new T.VideoFrameTexture(); texture.colorSpace = T.SRGBColorSpace; texture.setFrame(frame);
  t.after(() => texture.dispose()); return {texture, frame};
}
const draws = f => f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
const entries = draw => [...draw.groups.values()].flatMap(({group}) => group.entries);
const view = draw => entries(draw).find(entry => entry.resource?.texture).resource;
const uploaded = (f, texture) => f.copies.find(copy => copy.destination.texture === texture);
const sample = (f, texture) => { f.material.colorNode = N.texture(texture).rgb; };
function assertSource(f, descriptor) {
  const source = new T.WGSLNodeBuilder(f.object, f.renderer); source.scene = f.scene; source.camera = f.camera; source.build();
  const canonical = shader => shader.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
  assert.equal(canonical(descriptor.vertex), canonical(source.vertexShader));
  assert.equal(canonical(descriptor.fragment), canonical(source.fragmentShader));
}

for (const kind of ['video', 'frame']) test(`native ${kind} compiles without media capture and preserves source 2D WGSL and upload metadata`, async t => {
  const f = fixture(t), texture = kind === 'video' ? video(t, f) : frameTexture(t, f).texture;
  texture.premultiplyAlpha = true; texture.wrapS = T.RepeatWrapping; sample(f, texture);
  const before = f.host.owned.length, version = texture.version, image = texture.image;
  await f.renderer.compileAsync(f.scene, f.camera);
  assert.equal(f.host.owned.length, before); assert.equal(f.copies.length, 0); assert.equal(texture.version, version);
  await f.renderer.renderAsync(f.scene, f.camera);
  const draw = draws(f)[0], descriptor = f.captures.find(frame => frame.draws.length).draws[0].binding.nodeProgram;
  assertSource(f, descriptor); assert.match(descriptor.fragment, /texture_2d<f32>/); assert.doesNotMatch(descriptor.fragment, /texture_external/);
  assert.equal(view(draw).format, 'rgba8unorm-srgb'); assert.equal(view(draw).texture.width, 4); assert.equal(view(draw).texture.height, 2);
  const copy = uploaded(f, view(draw).texture); assert.notEqual(copy.source.source, image); assert.ok(copy.pixels.every(v => v === 17));
  assert.equal(copy.source.flipY, true); assert.equal(copy.destination.premultipliedAlpha, true); assert.equal(copy.destination.colorSpace, 'srgb');
  assert.equal(texture.image, image); assert.equal(texture.version, version); assert.equal(texture.isVideoTexture, true);
});

for (const kind of ['video', 'frame']) test(`ordinary ${kind === 'video' ? 'MeshBasicMaterial video map' : 'MeshStandardMaterial frame alphaMap'} selects native immutable media without authored nodes`, async t => {
  const f = fixture(t, {}, kind === 'video' ? T.MeshBasicMaterial : T.MeshStandardMaterial);
  const initial = kind === 'frame' ? frameTexture(t, f) : null;
  const texture = initial?.texture ?? video(t, f), slot = kind === 'video' ? 'map' : 'alphaMap';
  f.material[slot] = texture;
  assert.equal(f.material.isNodeMaterial, undefined); assert.equal(hasThreeNodeProgram(f.material), false);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  if (initial) initial.frame.close(); else texture.image.pixels.fill(53);
  f.renderer.render(f.scene, f.camera);
  let next;
  if (initial) {
    next = new f.host.VideoFrame(new Uint8Array(32).fill(91), {codedWidth: 4, codedHeight: 2}); texture.setFrame(next);
  } else texture.image.frame(91);
  f.renderer.render(f.scene, f.camera);
  if (next) next.close(); else texture.image.pixels.fill(127);
  await f.renderer.waitForGPU();
  const rendered = draws(f); assert.equal(rendered.length, 3);
  assert.deepEqual(rendered.map(draw => uploaded(f, view(draw).texture).pixels[0]), [17, 17, 91]);
  assert.equal(view(rendered[0]).texture, view(rendered[1]).texture);
  assert.notEqual(view(rendered[1]).texture, view(rendered[2]).texture);
  assert.ok(f.captures.every(frame => frame.draws.every(draw => draw.binding.nodeProgram)));
  assert.equal(f.material[slot], texture); assert.equal(hasThreeNodeProgram(f.material), false);
  if (initial) { assert.equal(initial.frame.closeCalls, 1); assert.equal(next.closeCalls, 1); }
});

test('queued VideoFrameTexture versions survive public close and never close or replace caller frames', async t => {
  const f = fixture(t), {texture, frame: first} = frameTexture(t, f), updates = [];
  texture.onUpdate = value => updates.push(value); sample(f, texture); await f.renderer.init();
  f.renderer.render(f.scene, f.camera); first.close();
  f.renderer.render(f.scene, f.camera);
  const second = new f.host.VideoFrame(new Uint8Array(32).fill(73), {codedWidth: 4, codedHeight: 2}); texture.setFrame(second);
  f.renderer.render(f.scene, f.camera); second.close(); await f.renderer.waitForGPU();
  const native = draws(f).map(draw => view(draw).texture); assert.equal(native[0], native[1]); assert.notEqual(native[1], native[2]);
  assert.ok(uploaded(f, native[0]).pixels.every(v => v === 17)); assert.ok(uploaded(f, native[2]).pixels.every(v => v === 73));
  assert.deepEqual(updates, [texture, texture]); assert.equal(first.closeCalls, 1); assert.equal(second.closeCalls, 1);
  assert.equal(texture.image, second); f.renderer.dispose();
  assert.ok(f.host.owned.every(value => value.closeCalls === 1)); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('VideoFrameTexture aliases keep independent requested frames and upload callbacks', async t => {
  const f = fixture(t), {texture: a, frame: first} = frameTexture(t, f), b = a.clone(), updates = [];
  t.after(() => b.dispose()); a.onUpdate = () => updates.push('a'); b.onUpdate = () => updates.push('b');
  assert.equal(a.source, b.source); assert.equal(a.version, b.version);
  await f.renderer.init(); sample(f, a); f.renderer.render(f.scene, f.camera);
  sample(f, b); f.renderer.render(f.scene, f.camera); first.close();
  const second = new f.host.VideoFrame(new Uint8Array(32).fill(73), {codedWidth: 4, codedHeight: 2}); b.setFrame(second);
  f.renderer.render(f.scene, f.camera); second.close(); sample(f, a); f.renderer.render(f.scene, f.camera);
  await f.renderer.waitForGPU();
  const native = draws(f).map(draw => view(draw).texture);
  assert.deepEqual(native.map(texture => uploaded(f, texture).pixels[0]), [17, 17, 73, 17]);
  assert.notEqual(native[0], native[1]); assert.equal(native[0], native[3]);
  assert.deepEqual(updates, ['a', 'b', 'b']); assert.deepEqual([a.version, b.version], [1, 2]);
  assert.equal(a.source, b.source); f.renderer.dispose();
  assert.ok(f.host.owned.every(frame => frame.closeCalls === 1));
});

test('shared media bindings refresh only from the same texture accepted version', async t => {
  const f = fixture(t), {texture: a, frame: first} = frameTexture(t, f), updates = [];
  a.onUpdate = () => updates.push('a'); await f.renderer.init();
  const owner = f.renderer._getSampledTextureInputs(), bind = texture => ({texture, isSampledTexture: true, isSampler: true});
  const old = owner.captureBinding(bind(a)); assert.equal(owner.resolve(old).texture.pixels[0], 17); first.close();
  const next = new f.host.VideoFrame(new Uint8Array(32).fill(91), {codedWidth: 4, codedHeight: 2}); a.setFrame(next);
  const requested = owner.captureBinding(bind(a)), refreshed = owner.captureShared(old); next.close();
  assert.ok(refreshed); assert.equal(owner.resolve(refreshed), owner.resolve(requested));
  assert.equal(owner.resolve(refreshed).texture.pixels[0], 91);
  const beforeAlias = owner.diagnostics.snapshotBytes, failure = new Error('application upload callback');
  const b = a.clone(), other = new f.host.VideoFrame(new Uint8Array(32).fill(73), {codedWidth: 4, codedHeight: 2});
  t.after(() => b.dispose()); b.onUpdate = () => { updates.push('b'); throw failure; }; b.setFrame(other);
  const alias = owner.captureBinding(bind(b)); other.close();
  assert.throws(() => owner.captureDispatch(() => owner.resolve(alias)), error => error === failure);
  assert.equal(owner.resolve(alias).texture.pixels[0], 73);
  const retained = owner.captureShared(old); assert.equal(owner.resolve(retained), owner.resolve(requested));
  assert.equal(owner.resolve(old).texture.pixels[0], 17); assert.equal(owner.captureShared(requested), null);
  assert.deepEqual(updates, ['a', 'a', 'b']);
  b.dispose(); assert.equal(owner.diagnostics.snapshotBytes, beforeAlias);
  for (const lease of [old, requested, refreshed, alias, retained]) owner.release(lease);
  a.dispose(); b.dispose(); assert.equal(owner.diagnostics.snapshotBytes, 0);
  assert.ok(f.host.owned.every(frame => frame.closeCalls === 1));
});

for (const callbacks of [true, false]) test(`native video upload requests preserve ${callbacks ? 'rVFC' : 'manual'} versions without calling fallback updates`, async t => {
  const f = fixture(t), texture = video(t, f, {callbacks}), image = texture.image; sample(f, texture); await f.renderer.init();
  const requested = image.requests; f.renderer.render(f.scene, f.camera);
  image.pixels.fill(53); image.currentTime = 3; f.renderer.render(f.scene, f.camera);
  image.pixels.fill(91); if (callbacks) image.frame(91); else texture.needsUpdate = true;
  f.renderer.render(f.scene, f.camera); image.pixels.fill(127); await f.renderer.waitForGPU();
  const native = draws(f).map(draw => view(draw).texture);
  assert.equal(native[0], native[1]); assert.notEqual(native[1], native[2]);
  assert.ok(uploaded(f, native[0]).pixels.every(v => v === 17)); assert.ok(uploaded(f, native[2]).pixels.every(v => v === 91));
  assert.equal(texture.version, 2); assert.equal(image.requests, requested + Number(callbacks)); assert.equal(image.cancellations, 0);
  assert.equal(image.paused, true); assert.equal(image.playbackRate, 1);
});

test('version-zero native media uses the pinned 1x1 default until its first upload request', async t => {
  const f = fixture(t), texture = video(t, f, {requested: false}); sample(f, texture); await f.renderer.init();
  f.renderer.render(f.scene, f.camera); texture.image.frame(81); f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const native = draws(f).map(draw => view(draw).texture);
  assert.deepEqual(native.map(value => [value.width, value.height]), [[1, 1], [4, 2]]);
  assert.equal(uploaded(f, native[0]), undefined); assert.equal(f.copies.length, 1); assert.ok(f.copies[0].pixels.every(v => v === 81));
});

test('a temporarily starved native video retains its captured image across acknowledged requests', async t => {
  const f = fixture(t), texture = video(t, f), image = texture.image; sample(f, texture); await f.renderer.init();
  f.renderer.render(f.scene, f.camera); image.readyState = 1; image.pixels.fill(67); texture.needsUpdate = true;
  f.renderer.render(f.scene, f.camera); image.readyState = 2; image.frame(93); f.renderer.render(f.scene, f.camera);
  await f.renderer.waitForGPU();
  assert.deepEqual(draws(f).map(draw => uploaded(f, view(draw).texture).pixels[0]), [17, 17, 93]);
});

test('HTML video without WebCodecs snapshots at call time and waits only for bitmap readiness', async t => {
  const f = fixture(t, {frames: false, defer: true}), texture = video(t, f); sample(f, texture); await f.renderer.init();
  f.renderer.render(f.scene, f.camera); texture.image.pixels.fill(53); texture.wrapS = T.RepeatWrapping;
  f.renderer.render(f.scene, f.camera); texture.image.frame(91); f.renderer.render(f.scene, f.camera);
  texture.image.pixels.fill(127); assert.equal(f.host.pending.length, 2); assert.equal(f.copies.length, 0);
  f.host.settle(); await f.renderer.waitForGPU();
  assert.deepEqual(draws(f).map(draw => uploaded(f, view(draw).texture).pixels[0]), [17, 17, 91]);
  assert.deepEqual(draws(f).map(draw => entries(draw).find(entry => entry.resource?.addressModeU).resource.addressModeU),
    ['clamp-to-edge', 'repeat', 'repeat']);
  f.renderer.dispose(); assert.ok(f.host.bitmaps.every(image => image.closeCalls === 1));
});

test('starved requests retain the previous immutable video while its bitmap is still preparing', async t => {
  const f = fixture(t, {frames: false, defer: true}), texture = video(t, f), updates = [];
  texture.onUpdate = () => updates.push(texture); sample(f, texture); await f.renderer.init();
  f.renderer.render(f.scene, f.camera);
  texture.image.readyState = 1; texture.image.pixels.fill(67); texture.needsUpdate = true;
  f.renderer.render(f.scene, f.camera); assert.equal(f.host.pending.length, 1);
  f.host.settle(); await f.renderer.waitForGPU();
  assert.deepEqual(draws(f).map(draw => uploaded(f, view(draw).texture).pixels[0]), [17, 17]);
  assert.deepEqual(updates, [texture, texture]); f.renderer.dispose();
  assert.ok(f.host.bitmaps.every(image => image.closeCalls === 1));
});

test('rejected capture rolls back owned frames without closing the application source', async t => {
  const f = fixture(t), {texture, frame} = frameTexture(t, f); await f.renderer.init();
  const owner = f.renderer._getSampledTextureInputs(), sentinel = new Error('rejected queued draw');
  assert.throws(() => owner.captureDispatch(() => {
    owner.captureBinding({texture, isSampledTexture: true, isSampler: true}); throw sentinel;
  }), error => error === sentinel);
  assert.equal(frame.closeCalls, 0); assert.equal(f.host.owned[1].closeCalls, 1); assert.equal(owner.diagnostics.snapshotBytes, 0);
});

test('disposing a renderer closes late bitmap captures exactly once without touching playback', async t => {
  const f = fixture(t, {frames: false, defer: true}), texture = video(t, f); sample(f, texture); await f.renderer.init();
  f.renderer.render(f.scene, f.camera); const idle = f.renderer.waitForGPU(); f.renderer.dispose();
  f.host.settle(); await assert.rejects(idle); await Promise.resolve();
  assert.ok(f.host.bitmaps.every(image => image.closeCalls === 1)); assert.equal(texture.image.cancellations, 0); assert.equal(f.copies.length, 0);
});

for (const frames of [true, false]) test(`native ${frames ? 'VideoFrame' : 'ImageBitmap'} capture preserves cross-origin refusal`, async t => {
  const f = fixture(t, {frames}), texture = video(t, f); texture.image.tainted = true; sample(f, texture); await f.renderer.init();
  const owner = f.renderer._getSampledTextureInputs();
  await assert.rejects(f.renderer.renderAsync(f.scene, f.camera), {name: 'SecurityError'});
  assert.equal(f.copies.length, 0); assert.equal(draws(f).length, 0);
  texture.dispose(); assert.equal(owner.diagnostics.snapshotBytes, 0);
});

test('a VideoFrameTexture flag cannot disguise an ordinary data texture', async t => {
  const f = fixture(t), texture = new T.DataTexture(new Uint8Array([17, 17, 17, 255]), 1, 1);
  texture.isVideoFrameTexture = true; texture.needsUpdate = true; sample(f, texture);
  t.after(() => texture.dispose()); await f.renderer.init();
  await assert.rejects(f.renderer.renderAsync(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_TEXTURE'});
  assert.equal(f.copies.length, 0); assert.equal(draws(f).length, 0); assert.equal(f.host.owned.length, 0);
});
