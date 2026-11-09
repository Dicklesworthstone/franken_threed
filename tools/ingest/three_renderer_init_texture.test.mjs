/** Public texture preloading over the production native renderer. The existing
 * device seam records uploads, draws and dispatch bindings; it does not run WGSL. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import SourceTextures from '../../upstream/three.js/src/renderers/common/Textures.js';
import SourceTextureUtils from '../../upstream/three.js/src/renderers/webgpu/utils/WebGPUTextureUtils.js';

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
    destination.texture.uploads.push(upload); destination.texture.pixels = upload.bytes.slice(); uploads.push(upload);
  };
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  device.createComputePipeline = device.createRenderPipeline;
  device.createComputePipelineAsync = async descriptor => device.createComputePipeline(descriptor);
  const encode = device.createCommandEncoder;
  device.createCommandEncoder = () => {
    const encoder = encode(), passes = [], copies = [];
    return {...encoder, beginComputePass(descriptor) {
      const pass = {descriptor, draws: [], groups: new Map(), dispatches: []}; passes.push(pass); computes.push(pass);
      return {setPipeline(value) {pass.pipeline = value;}, setBindGroup(index, group) {pass.groups.set(index, group);},
        dispatchWorkgroups(...size) {pass.dispatches.push(size);}, end() {}};
    }, copyTextureToTexture(source, destination, size) {copies.push({source, destination, size});},
    finish() {return [...encoder.finish(), ...passes, ...(copies.length ? [{draws: [], copies}] : [])];}};
  };
  const submit = device.queue.submit;
  device.queue.submit = commands => {
    for (const passes of commands) for (const pass of passes) for (const {source, destination} of pass.copies ?? [])
      destination.texture.pixels = source.texture.pixels?.slice();
    submit(commands);
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

test('pending mutable-image captures fail synchronously without publishing upload history', async t => {
  const f = fixture(t); await f.renderer.init();
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

function sourceTextureOwner(t, device) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'GPUTextureUsage');
  Object.defineProperty(globalThis, 'GPUTextureUsage', {configurable: true, value: {COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16}});
  t.after(() => {if (previous) Object.defineProperty(globalThis, 'GPUTextureUsage', previous); else delete globalThis.GPUTextureUsage;});
  const data = new WeakMap(), operations = [];
  const backend = {device, get(texture) {let value = data.get(texture); if (!value) data.set(texture, value = {}); return value;},
    delete: texture => data.delete(texture), utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})}};
  const gpu = new SourceTextureUtils(backend);
  Object.assign(backend, {createDefaultTexture(texture) {gpu.createDefaultTexture(texture); operations.push('default');},
    createTexture(texture, options) {gpu.createTexture(texture, options); operations.push('allocate');},
    updateTexture() {operations.push('upload');}, generateMipmaps() {operations.push('mips');},
    destroyTexture: (texture, isDefault) => gpu.destroyTexture(texture, isDefault)});
  const owner = new SourceTextures({getRenderTarget: () => null}, backend, {createTexture() {}, destroyTexture() {}});
  return {update: texture => owner.updateTexture(texture), view: texture => backend.get(texture).texture, operations};
}

test('native version-zero placeholders match source allocation formats without reading CPU pixels', async t => {
  const f = fixture(t); await f.renderer.init(); const source = sourceTextureOwner(t, f.device);
  for (const [format, type] of [[T.RGBAFormat, T.UnsignedByteType], [T.RedFormat, T.HalfFloatType],
    [T.RGIntegerFormat, T.IntType], [T.RGBAIntegerFormat, T.UnsignedIntType]]) {
    const input = new T.DataTexture(null, 8192, 8192, format, type), oracle = new T.DataTexture(null, 8192, 8192, format, type);
    let callbacks = 0; input.onUpdate = () => {callbacks++;};
    source.update(oracle); f.renderer.initTexture(input);
    const actual = view(f.renderer, input).texture, expected = source.view(oracle);
    assert.deepEqual([actual.width, actual.height, actual.format], [expected.width, expected.height, expected.format]);
    assert.deepEqual([actual.width, actual.height], [1, 1]); assert.equal(callbacks, 0);
    assert.equal(f.renderer._getSampledTextureInputs().diagnostics.snapshotBytes, 0);
    input.image = {width: 2, height: 2, data: new Uint8Array(16).fill(93)};
    f.renderer.initTexture(input); assert.equal(view(f.renderer, input).texture, actual);
    input.dispose(); oracle.dispose();
  }
  const empty = new T.Texture(); f.renderer.initTexture(empty);
  assert.equal(view(f.renderer, empty).texture.width, 1); empty.dispose();
  const color = new T.Texture(), expectedColor = new T.Texture();
  color.colorSpace = expectedColor.colorSpace = T.SRGBColorSpace;
  source.update(expectedColor); f.renderer.initTexture(color);
  assert.equal(view(f.renderer, color).texture.format, source.view(expectedColor).format);
  color.dispose(); expectedColor.dispose();
  assert.equal(f.uploads.length, 0);
});

test('native no-data allocation follows source callbacks and waits for another requested version', async t => {
  const f = fixture(t); await f.renderer.init(); const source = sourceTextureOwner(t, f.device);
  const input = data(t, 37), oracle = new T.DataTexture(new Uint8Array(16).fill(37), 2, 2);
  oracle.needsUpdate = true; input.source.dataReady = oracle.source.dataReady = false;
  const events = [], expected = []; input.onUpdate = texture => {assert.equal(texture, input); events.push('updated');};
  oracle.onUpdate = () => expected.push('updated');
  source.update(oracle); f.renderer.initTexture(input);
  const allocated = view(f.renderer, input).texture;
  assert.deepEqual([allocated.width, allocated.height, allocated.format], [source.view(oracle).width, source.view(oracle).height, source.view(oracle).format]);
  assert.deepEqual(events, expected); assert.equal(f.uploads.length, 0);
  input.source.dataReady = oracle.source.dataReady = true;
  source.update(oracle); f.renderer.initTexture(input);
  assert.equal(view(f.renderer, input).texture, allocated); assert.equal(f.uploads.length, 0); assert.deepEqual(events, expected);
  input.needsUpdate = oracle.needsUpdate = true;
  source.update(oracle); f.renderer.initTexture(input);
  assert.equal(f.uploads.length, 1); assert.deepEqual([...view(f.renderer, input).texture.pixels], Array(16).fill(37));
  assert.deepEqual(events, expected); oracle.dispose();
});

test('native no-data extents use source width/height fallback without decoded image bytes', async t => {
  const f = fixture(t); await f.renderer.init(); const source = sourceTextureOwner(t, f.device);
  for (const image of [{width: 0, height: 0, data: null}, {width: 3, height: 2, naturalWidth: 19, naturalHeight: 13}]) {
    const input = new T.Texture(image), oracle = new T.Texture(image);
    input.source.dataReady = oracle.source.dataReady = false; input.needsUpdate = oracle.needsUpdate = true;
    source.update(oracle); f.renderer.initTexture(input);
    const actual = view(f.renderer, input).texture, expected = source.view(oracle);
    assert.deepEqual([actual.width, actual.height], [expected.width, expected.height]);
    input.dispose(); oracle.dispose();
  }
  assert.equal(f.uploads.length, 0);
});

test('native no-data versions preserve earlier GPU pixels and queued texture generations', async t => {
  const f = fixture(t), input = data(t, 29); await f.renderer.init();
  const owner = f.renderer._getSampledTextureInputs(), first = owner.captureBinding({texture: input, isSampledTexture: true});
  assert.equal(f.uploads.length, 0, 'the earlier accepted draw has not prepared its texture yet');
  input.image.data.fill(91); input.source.dataReady = false; input.needsUpdate = true;
  f.renderer.initTexture(input); const deferred = view(f.renderer, input).texture;
  const before = owner.resolve(first).texture;
  assert.notEqual(deferred, before); assert.deepEqual([...deferred.pixels], Array(16).fill(29));
  assert.deepEqual([...owner.resolve(first).texture.pixels], Array(16).fill(29));
  input.source.dataReady = true; f.renderer.initTexture(input); assert.equal(view(f.renderer, input).texture, deferred);
  input.needsUpdate = true; f.renderer.initTexture(input);
  assert.deepEqual([...view(f.renderer, input).texture.pixels], Array(16).fill(91));
  assert.equal(f.uploads.length, 2); owner.release(first);
  input.dispose(); assert.equal(owner.diagnostics.snapshotBytes, 0);
  input.source.dataReady = false; f.renderer.initTexture(input);
  assert.equal(view(f.renderer, input).texture.pixels, undefined, 'a new no-data residency starts at WebGPU zero initialization');
});

test('native default and no-data texture aliases keep independent acknowledgement and disposal', async t => {
  const f = fixture(t), first = new T.Texture(); await f.renderer.init();
  const alias = first.clone(); alias.version = 0;
  f.renderer.initTexture(first); f.renderer.initTexture(alias);
  const originalDefault = view(f.renderer, first).texture;
  alias.dispose(); assert.equal(originalDefault.destroyed, undefined);
  const input = data(t, 51), sibling = input.clone(); input.source.dataReady = false;
  let a = 0, b = 0; input.onUpdate = () => {a++;}; sibling.onUpdate = () => {b++;};
  f.renderer.initTexture(input); f.renderer.initTexture(sibling);
  assert.deepEqual([a, b], [1, 1]); assert.equal(f.uploads.length, 0);
  const retained = view(f.renderer, sibling).texture;
  input.source.dataReady = true; input.needsUpdate = true; f.renderer.initTexture(input);
  assert.equal(view(f.renderer, sibling).texture, retained); assert.equal(retained.pixels, undefined);
  input.dispose(); assert.equal(retained.destroyed, undefined);
  sibling.dispose(); first.dispose();
});
