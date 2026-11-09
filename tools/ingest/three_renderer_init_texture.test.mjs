/** Public texture preloading over the production native renderer. The existing
 * device seam records uploads, draws and dispatch bindings; it does not run WGSL. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T);
const deferred = () => { let resolve; const promise = new Promise(yes => {resolve = yes;}); return {promise, resolve}; };
const drawViews = draw => [...draw.groups.values()].flatMap(({group}) => group.entries)
  .map(entry => entry.resource).filter(resource => resource?.texture);
function fixture(t) {
  const device = geometryDevice(), textures = [], uploads = [], computes = [];
  device.features = new Set();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureDimension3D: 256,
    maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1, uploads: [],
      createView(options = {}) {return {texture: this, ...options};}, destroy() {this.destroyed = true;}};
    textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = (destination, data, layout, size) => {
    const upload = {destination, bytes: new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice(), layout, size};
    destination.texture.uploads.push(upload); uploads.push(upload);
  };
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  device.createComputePipeline = device.createRenderPipeline;
  device.createComputePipelineAsync = async descriptor => device.createComputePipeline(descriptor);
  const encode = device.createCommandEncoder;
  device.createCommandEncoder = () => {
    const encoder = encode(), passes = [];
    return {...encoder, beginComputePass(descriptor) {
      const pass = {descriptor, draws: [], groups: new Map(), dispatches: []}; passes.push(pass); computes.push(pass);
      return {setPipeline(value) {pass.pipeline = value;}, setBindGroup(index, group) {pass.groups.set(index, group);},
        dispatchWorkgroups(...size) {pass.dispatches.push(size);}, end() {}};
    }, finish() {return [...encoder.finish(), ...passes];}};
  };
  let configuration;
  const canvas = {width: 16, height: 16, style: {}, getContext: () => ({configure(value) {configuration = value;},
    unconfigure() {}, getCurrentTexture: () => device.createTexture({size: [16, 16], format: configuration.format, usage: 16})})};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 10); camera.position.z = 3;
  const geometry = new T.PlaneGeometry(), material = new T.MeshBasicNodeMaterial(), mesh = new T.Mesh(geometry, material);
  mesh.frustumCulled = false; scene.add(mesh);
  t.after(async () => {await renderer.dispose(); geometry.dispose(); material.dispose();});
  return {renderer, device, textures, uploads, computes, scene, camera, material};
}
function data(t, value = 17) {
  const texture = new T.DataTexture(new Uint8Array(16).fill(value), 2, 2);
  texture.needsUpdate = true; t.after(() => texture.dispose()); return texture;
}
function view(renderer, texture) {
  const owner = renderer._getSampledTextureInputs(), lease = owner.captureBinding({texture, isSampledTexture: true});
  try {return owner.withDispatch([lease], () => owner.resolve(lease));} finally {owner.release(lease);}
}

test('initTexture preserves the synchronous initialized prerequisite and upload callback boundary', async t => {
  const f = fixture(t), texture = data(t), events = [];
  let sourceError;
  try {T.Renderer.prototype.initTexture.call({_initialized: false}, texture);} catch (error) {sourceError = error;}
  assert.throws(() => f.renderer.initTexture(texture), error => error.constructor === sourceError.constructor && error.message === sourceError.message);
  assert.equal(f.uploads.length, 0);
  await f.renderer.init();
  texture.onUpdate = function (received) {
    events.push('updated'); assert.equal(this, texture); assert.equal(received, texture);
    assert.equal(f.uploads.length, 1, 'the upload precedes the source callback');
  };
  events.push('before'); assert.equal(f.renderer.initTexture(texture), undefined); events.push('after');
  assert.deepEqual(events, ['before', 'updated', 'after']);
  const first = view(f.renderer, texture);
  texture.image.data.fill(91);
  assert.equal(f.renderer.initTexture(texture), undefined);
  assert.equal(view(f.renderer, texture), first); assert.equal(f.uploads.length, 1);
  assert.deepEqual([...first.texture.uploads[0].bytes], Array(16).fill(17));
  assert.deepEqual(events, ['before', 'updated', 'after']);
});

test('preloaded texture history and residency feed both native graphics and compute', async t => {
  const f = fixture(t), texture = data(t, 39);
  await f.renderer.init(); f.renderer.initTexture(texture);
  const first = view(f.renderer, texture); texture.image.data.fill(201);
  f.material.colorNode = N.texture(texture).rgb;
  await f.renderer.renderAsync(f.scene, f.camera);
  const draw = f.device.snapshots.flat().find(draw => draw.pipeline.vertex.entryPoint === 'main' && draw.streams.size);
  assert.ok(drawViews(draw).some(value => value.texture === first.texture));
  const values = new T.StorageBufferAttribute(new Float32Array(4), 4), output = N.storage(values, 'vec4', 1);
  const kernel = N.Fn(() => {output.element(0).assign(N.textureLoad(texture, N.ivec2(0)));})().compute(1);
  t.after(() => kernel.dispose());
  await f.renderer.computeAsync(kernel); await f.renderer.waitForGPU();
  const sampled = [...f.computes.at(-1).groups.values()].flatMap(group => group.entries).map(entry => entry.resource);
  assert.ok(sampled.some(value => value?.texture === first.texture));
  assert.equal(f.uploads.length, 1); assert.deepEqual([...f.uploads[0].bytes], Array(16).fill(39));
});

test('preloading during queued draw preparation keeps earlier and later accepted upload versions', async t => {
  const f = fixture(t), texture = data(t, 23), gate = deferred(), started = deferred();
  await f.renderer.init(); f.material.colorNode = N.texture(texture).rgb;
  const pipeline = f.device.createRenderPipelineAsync.bind(f.device);
  f.device.createRenderPipelineAsync = async descriptor => {started.resolve(); await gate.promise; return pipeline(descriptor);};
  const firstDraw = f.renderer.renderAsync(f.scene, f.camera); await started.promise;
  const events = []; texture.image.data.fill(73); texture.needsUpdate = true;
  texture.onUpdate = () => events.push('updated');
  assert.equal(f.renderer.initTexture(texture), undefined); events.push('returned');
  assert.deepEqual(events, ['updated', 'returned']);
  const preloaded = view(f.renderer, texture);
  const secondDraw = f.renderer.renderAsync(f.scene, f.camera); texture.image.data.fill(149);
  gate.resolve(); await Promise.all([firstDraw, secondDraw]);
  const draws = f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main' && draw.streams.size);
  assert.equal(draws.length, 2);
  const selected = draws.map(draw => drawViews(draw).find(value => value.texture.uploads.length)?.texture);
  assert.notEqual(selected[0], selected[1]); assert.equal(selected[1], preloaded.texture);
  assert.deepEqual(selected.map(texture => [...texture.uploads.at(-1).bytes]), [Array(16).fill(23), Array(16).fill(73)]);
});

test('initTextureAsync retains the deprecated alias and reads input after initialization', async t => {
  const f = fixture(t), texture = data(t, 11), gate = deferred(), warnings = [];
  t.mock.method(console, 'warn', (...args) => warnings.push(args));
  const init = f.renderer.init.bind(f.renderer);
  f.renderer.init = async () => {await gate.promise; return init();};
  const pending = f.renderer.initTextureAsync(texture);
  assert.ok(pending instanceof Promise); assert.equal(f.uploads.length, 0);
  texture.image.data.fill(81); gate.resolve(); assert.equal(await pending, undefined);
  assert.deepEqual([...f.uploads[0].bytes], Array(16).fill(81));
  await f.renderer.initTextureAsync(texture);
  assert.equal(f.uploads.length, 1); assert.equal(warnings.length, 1);
  assert.match(warnings[0].join(' '), /initTextureAsync\(\).*deprecated.*initTexture\(\)/);
});

test('storage, target, depth and framebuffer preloads use their existing allocation owners', async t => {
  const f = fixture(t); await f.renderer.init();
  const target = new T.RenderTarget(2, 2, {depthBuffer: false}), storage = new T.StorageTexture(2, 2);
  const depth = new T.DepthTexture(2, 2, T.UnsignedIntType), framebuffer = new T.FramebufferTexture(2, 2);
  framebuffer.type = T.HalfFloatType;
  t.after(() => {target.dispose(); storage.dispose(); depth.dispose(); framebuffer.dispose();});
  for (const texture of [storage, target.texture, depth, framebuffer]) {
    let callbacks = 0; texture.onUpdate = () => {callbacks++;};
    assert.equal(f.renderer.initTexture(texture), undefined);
    const first = view(f.renderer, texture), allocations = f.textures.length;
    f.renderer.initTexture(texture); assert.equal(view(f.renderer, texture), first);
    assert.equal(f.textures.length, allocations); assert.equal(callbacks, 0);
  }
  assert.equal(framebuffer.type, T.UnsignedByteType); assert.equal(f.uploads.length, 0);
});

test('requested versions upload again and disposal releases preloaded residency', async t => {
  const f = fixture(t), texture = data(t, 31); await f.renderer.init();
  f.renderer.initTexture(texture); const first = view(f.renderer, texture).texture;
  texture.image.data.fill(47); texture.needsUpdate = true;
  f.renderer.initTexture(texture); const second = view(f.renderer, texture).texture;
  assert.notEqual(second, first); assert.deepEqual([...second.uploads[0].bytes], Array(16).fill(47));
  texture.dispose(); await f.renderer.waitForGPU(); assert.equal(second.destroyed, true);
  texture.image.data.fill(59); f.renderer.initTexture(texture);
  const reused = view(f.renderer, texture).texture;
  assert.notEqual(reused, second); assert.deepEqual([...reused.uploads[0].bytes], Array(16).fill(59));
});

test('native data preloads keep full two-dimensional uploads and accepted partial array layers', async t => {
  const f = fixture(t), ordinary = data(t, 13); await f.renderer.init();
  f.renderer.initTexture(ordinary);
  ordinary.image.data.fill(41); ordinary.addUpdateRange(0, 4); ordinary.needsUpdate = true;
  f.renderer.initTexture(ordinary);
  assert.deepEqual([...view(f.renderer, ordinary).texture.uploads[0].bytes], Array(16).fill(41));
  assert.deepEqual(ordinary.updateRanges, [{start: 0, count: 4}], 'native 2D upload does not consume WebGL row ranges');
  const array = new T.DataArrayTexture(new Uint8Array(32).fill(19), 2, 2, 2);
  array.needsUpdate = true; t.after(() => array.dispose()); f.renderer.initTexture(array);
  array.image.data.fill(83); array.addLayerUpdate(1); array.needsUpdate = true;
  f.renderer.initTexture(array);
  const bytes = view(f.renderer, array).texture.uploads[0].bytes;
  assert.deepEqual([...bytes], [...Array(16).fill(19), ...Array(16).fill(83)]);
  assert.equal(array.layerUpdates.size, 0, 'the acknowledged source layer request is consumed');
});

test('unready defaults and pending mutable-image captures fail synchronously without publishing upload history', async t => {
  const f = fixture(t); await f.renderer.init();
  const unready = new T.DataTexture(new Uint8Array(16), 2, 2); t.after(() => unready.dispose());
  assert.throws(() => f.renderer.initTexture(unready), {code: 'THREE_TEXTURE_NOT_READY'});
  unready.needsUpdate = true; unready.source.dataReady = false;
  assert.throws(() => f.renderer.initTexture(unready), {code: 'THREE_TEXTURE_NOT_READY'});
  const gate = deferred(); let captures = 0, closed = 0;
  class Canvas {constructor() {this.width = this.height = 2;}}
  class Bitmap extends Canvas {close() {closed++;}}
  const replacements = {OffscreenCanvas: Canvas, ImageBitmap: Bitmap,
    createImageBitmap: () => {captures++; return gate.promise;}};
  for (const [name, value] of Object.entries(replacements)) {
    const previous = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, {configurable: true, writable: true, value});
    t.after(() => {if (previous) Object.defineProperty(globalThis, name, previous); else delete globalThis[name];});
  }
  const image = new Canvas(), texture = new T.CanvasTexture(image); t.after(() => texture.dispose());
  assert.throws(() => f.renderer.initTexture(texture), {code: 'THREE_COMPUTE_NOT_READY'});
  assert.equal(captures, 1); assert.equal(f.uploads.length, 0);
  assert.deepEqual(f.renderer._getSampledTextureInputs().diagnostics, {textures: 0, versions: 0, snapshotBytes: 0});
  gate.resolve(new Bitmap()); await gate.promise; await Promise.resolve();
  assert.equal(closed, 1, 'a rejected pending capture retires its private bitmap');
  assert.equal(texture.image, image, 'the application image is retained');
});
