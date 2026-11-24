/** Ordinary loading maps use the pinned native material/texture path from the
 * first draw. Recording devices prove source WGSL and accepted upload histories,
 * not rendered pixels or browser loader behavior. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import SourceTextures from '../../upstream/three.js/src/renderers/common/Textures.js';
import SourceTextureUtils from '../../upstream/three.js/src/renderers/webgpu/utils/WebGPUTextureUtils.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const Renderer = createWebGPURendererClass(T);
const deferred = () => {let resolve; const promise = new Promise(yes => {resolve = yes;}); return {promise, resolve};};
const draws = f => f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main' && draw.streams.size);
const views = draw => [...draw.groups.values()].flatMap(({group}) => group.entries)
  .map(entry => entry.resource).filter(resource => resource?.texture);
const sampled = draw => views(draw).find(view => view.texture.format === 'rgba8unorm')?.texture;
const mapUploads = f => f.uploads.filter(upload => upload.destination.texture.format === 'rgba8unorm');

function fixture(t, Material = T.MeshStandardMaterial) {
  const device = geometryDevice(), textures = [], uploads = [], captures = [];
  device.features = new Set();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1, uploads: [],
      createView(options = {}) {return {texture: this, ...options};}, destroy() {this.destroyed = true;}};
    textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = (destination, data, layout, size) => {
    const bytes = new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice();
    const upload = {destination, bytes, layout, size};
    uploads.push(upload); destination.texture.uploads.push(upload); destination.texture.pixels = bytes.slice();
  };
  const encode = device.createCommandEncoder;
  device.createCommandEncoder = () => {
    const encoder = encode(), copies = [];
    return {...encoder, copyTextureToTexture(source, destination, size) {copies.push({source, destination, size});},
      finish() {return [...encoder.finish(), ...(copies.length ? [{draws: [], copies}] : [])];}};
  };
  const submit = device.queue.submit;
  device.queue.submit = commands => {
    for (const passes of commands) for (const pass of passes) for (const {source, destination} of pass.copies ?? [])
      destination.texture.pixels = source.texture.pixels?.slice();
    submit(commands);
  };
  let configuration;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) {configuration = value;}, unconfigure() {},
    getCurrentTexture: () => device.createTexture({size: [32, 32], format: configuration.format, usage: 16})})};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const material = new Material(), object = new T.Mesh(new T.PlaneGeometry(), material), scene = new T.Scene();
  object.frustumCulled = false; scene.add(object);
  const light = new T.DirectionalLight(0xffffff, 1); light.position.set(1, 2, 3); scene.add(light);
  const camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 4;
  const capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function (request) {const frame = capture.call(this, request); if (frame) captures.push(frame); return frame;};
  t.after(async () => {await renderer.dispose(); object.geometry.dispose(); material.dispose();});
  return {renderer, device, textures, uploads, captures, material, object, scene, camera};
}
function sourceTextures(t, device) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'GPUTextureUsage');
  Object.defineProperty(globalThis, 'GPUTextureUsage', {configurable: true, value: {
    COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16}});
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
  return {update: texture => owner.updateTexture(texture), allocation: texture => backend.get(texture).texture, operations};
}
function sourceShader(f, descriptor) {
  const builder = new T.WGSLNodeBuilder(f.object, f.renderer);
  builder.scene = f.scene; builder.camera = f.camera;
  builder.lightsNode = T.TSL.lights(f.material.isMeshBasicMaterial ? [] : f.scene.children.filter(object => object.isLight));
  builder.build();
  const canonical = shader => shader.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
  assert.equal(canonical(descriptor.vertex), canonical(builder.vertexShader));
  assert.equal(canonical(descriptor.fragment), canonical(builder.fragmentShader));
}
const descriptor = f => f.captures.findLast(frame => frame.draws.length).draws[0].binding.nodeProgram;

for (const Material of [T.MeshStandardMaterial, T.MeshPhysicalMaterial, T.MeshNormalMaterial, T.MeshMatcapMaterial,
  T.MeshStandardNodeMaterial, T.MeshNormalNodeMaterial])
test(`${Material.name} renders an unloaded object-space normal map with source WGSL and default storage`, async t => {
  const f = fixture(t, Material), input = new T.Texture(), oracle = new T.Texture();
  const originalSource = input.source, source = sourceTextures(t, f.device); let callbacks = 0;
  input.onUpdate = () => {callbacks++;}; f.material.normalMap = input; f.material.normalMapType = T.ObjectSpaceNormalMap;
  t.after(() => {input.dispose(); oracle.dispose();}); source.update(oracle);
  await f.renderer.renderAsync(f.scene, f.camera);
  const actual = sampled(draws(f)[0]), expected = source.allocation(oracle);
  assert.ok(actual, 'the normal map must be bound on the first draw');
  assert.deepEqual([actual.width, actual.height, actual.format], [expected.width, expected.height, expected.format]);
  assert.deepEqual([actual.width, actual.height], [1, 1]); assert.equal(mapUploads(f).length, 0); assert.equal(callbacks, 0);
  assert.equal(input.image, null); assert.equal(input.source, originalSource); assert.equal(input.version, 0);
  assert.equal(f.material.normalMap, input); sourceShader(f, descriptor(f));
});

test('an ordinary Phong material keeps nonuniform normal scale while its normal map is loading', async t => {
  const f = fixture(t, T.MeshPhongMaterial), input = new T.Texture(); t.after(() => input.dispose());
  f.material.normalMap = input; f.material.normalScale.set(.25, .75);
  await f.renderer.renderAsync(f.scene, f.camera);
  assert.deepEqual([sampled(draws(f)[0]).width, sampled(draws(f)[0]).height], [1, 1]);
  assert.deepEqual(f.material.normalScale.toArray(), [.25, .75]); sourceShader(f, descriptor(f));
});

for (const [Material, slot] of [[T.MeshBasicMaterial, 'map'], [T.MeshBasicMaterial, 'alphaMap'],
  [T.MeshLambertMaterial, 'emissiveMap'], [T.MeshMatcapMaterial, 'matcap'], [T.MeshToonMaterial, 'gradientMap']])
test(`an unloaded ${Material.name}.${slot} selects the source material path`, async t => {
  const f = fixture(t, Material), input = new T.Texture(); t.after(() => input.dispose()); f.material[slot] = input;
  await f.renderer.renderAsync(f.scene, f.camera);
  assert.ok(descriptor(f)); sourceShader(f, descriptor(f)); assert.equal(mapUploads(f).length, 0);
  assert.equal(f.material[slot], input); assert.equal(input.version, 0);
});

test('queued loading and loaded draws retain separate native generations without consuming unrequested pixels', async t => {
  const f = fixture(t), input = new T.DataTexture(null, 2, 2), originalSource = input.source;
  t.after(() => input.dispose()); f.material.normalMap = input; f.material.normalMapType = T.ObjectSpaceNormalMap;
  await f.renderer.init(); const started = deferred(), gate = deferred(), events = [];
  const pipeline = f.device.createRenderPipelineAsync.bind(f.device);
  f.device.createRenderPipelineAsync = async value => {started.resolve(); await gate.promise; return pipeline(value);};
  const first = f.renderer.renderAsync(f.scene, f.camera); await started.promise;
  const accepted = new Uint8Array(16).fill(23); input.image = {data: accepted, width: 2, height: 2}; input.needsUpdate = true;
  input.onUpdate = function (texture) {
    assert.equal(this, input); assert.equal(texture, input); assert.equal(mapUploads(f).length, 1); events.push('uploaded');
  };
  const second = f.renderer.renderAsync(f.scene, f.camera); accepted.fill(91);
  gate.resolve(); await Promise.all([first, second]);
  const [loading, loaded] = draws(f).map(sampled);
  assert.notEqual(loading, loaded); assert.deepEqual([loading.width, loading.height], [1, 1]);
  assert.equal(loading.pixels, undefined); assert.deepEqual([...loaded.pixels], Array(16).fill(23));
  assert.deepEqual(events, ['uploaded']); assert.equal(input.source, originalSource); assert.equal(input.image.data, accepted);
  assert.equal(f.captures[0].draws[0].binding.nodeProgram, f.captures[1].draws[0].binding.nodeProgram,
    'loader completion updates residency without replacing the material program');
  input.image = {data: new Uint8Array(16).fill(149), width: 2, height: 2};
  await f.renderer.renderAsync(f.scene, f.camera); assert.equal(sampled(draws(f).at(-1)), loaded);
  assert.equal(mapUploads(f).length, 1); assert.deepEqual(events, ['uploaded']);
  const replacementSource = new T.TextureSource({data: new Uint8Array(16).fill(197), width: 2, height: 2});
  input.source = replacementSource;
  input.onUpdate = () => events.push('replacement'); input.needsUpdate = true;
  await f.renderer.renderAsync(f.scene, f.camera); const replaced = sampled(draws(f).at(-1));
  assert.notEqual(replaced, loaded); assert.deepEqual([...replaced.pixels], Array(16).fill(197));
  assert.deepEqual(events, ['uploaded', 'replacement']); assert.equal(input.source, replacementSource);
  f.material.normalMap = null; f.material.needsUpdate = true;
  await f.renderer.renderAsync(f.scene, f.camera);
  assert.ok(descriptor(f), 'removing the loading map retains the selected native scene/material owner');
  assert.equal(sampled(draws(f).at(-1)), undefined);
  input.dispose(); await f.renderer.waitForGPU(); assert.equal(replaced.destroyed, true);
});

test('version-zero defaults survive unrequested image and Source replacement until loading is acknowledged', async t => {
  const f = fixture(t, T.MeshBasicMaterial), input = new T.DataTexture(null, 2, 2);
  t.after(() => input.dispose()); f.material.map = input; let callbacks = 0; input.onUpdate = () => {callbacks++;};
  await f.renderer.renderAsync(f.scene, f.camera); const initial = sampled(draws(f)[0]);
  input.image = {data: new Uint8Array(16).fill(17), width: 2, height: 2};
  await f.renderer.renderAsync(f.scene, f.camera); assert.equal(sampled(draws(f).at(-1)), initial);
  const replacementSource = new T.TextureSource({data: new Uint8Array(16).fill(79), width: 2, height: 2});
  input.source = replacementSource;
  await f.renderer.renderAsync(f.scene, f.camera); assert.equal(sampled(draws(f).at(-1)), initial);
  assert.equal(mapUploads(f).length, 0); assert.equal(callbacks, 0); assert.equal(input.version, 0);
  input.needsUpdate = true; await f.renderer.renderAsync(f.scene, f.camera);
  assert.deepEqual([...sampled(draws(f).at(-1)).pixels], Array(16).fill(79));
  assert.equal(input.source, replacementSource); assert.equal(callbacks, 1);
});

test('loaded ordinary maps match the source version rule across Source replacement, requested upload and disposal', async t => {
  const f = fixture(t, T.MeshBasicMaterial), input = new T.DataTexture(new Uint8Array(16).fill(29), 2, 2);
  const oracle = new T.DataTexture(new Uint8Array(16).fill(29), 2, 2), source = sourceTextures(t, f.device);
  t.after(() => {input.dispose(); oracle.dispose();}); f.material.map = input;
  const events = [], expectedEvents = []; input.onUpdate = () => events.push('update'); oracle.onUpdate = () => expectedEvents.push('update');
  input.needsUpdate = oracle.needsUpdate = true;
  source.update(oracle); await f.renderer.renderAsync(f.scene, f.camera); const initial = sampled(draws(f)[0]);
  assert.deepEqual([...initial.pixels], Array(16).fill(29));
  input.source = new T.TextureSource(null); oracle.source = new T.TextureSource(null);
  const replacement = input.source, originalVersion = input.version;
  source.update(oracle); await f.renderer.renderAsync(f.scene, f.camera);
  assert.equal(sampled(draws(f).at(-1)), initial, 'unrequested Source replacement cannot invalidate accepted GPU pixels');
  assert.equal(input.source, replacement); assert.equal(input.version, originalVersion);
  input.image = oracle.image = {data: new Uint8Array(16).fill(83), width: 2, height: 2};
  source.update(oracle); await f.renderer.renderAsync(f.scene, f.camera);
  assert.equal(sampled(draws(f).at(-1)), initial); assert.equal(mapUploads(f).length, 1);
  assert.deepEqual(events, expectedEvents); assert.deepEqual(source.operations, ['allocate', 'upload']);
  input.needsUpdate = oracle.needsUpdate = true;
  source.update(oracle); await f.renderer.renderAsync(f.scene, f.camera); const requested = sampled(draws(f).at(-1));
  assert.notEqual(requested, initial); assert.deepEqual([...requested.pixels], Array(16).fill(83));
  assert.deepEqual(events, expectedEvents); assert.deepEqual(events, ['update', 'update']);
  input.source = new T.TextureSource({data: new Uint8Array(16).fill(137), width: 2, height: 2});
  oracle.source = new T.TextureSource({data: new Uint8Array(16).fill(137), width: 2, height: 2});
  input.dispose(); oracle.dispose(); await f.renderer.waitForGPU(); assert.equal(requested.destroyed, true);
  source.update(oracle); await f.renderer.renderAsync(f.scene, f.camera);
  assert.deepEqual([...sampled(draws(f).at(-1)).pixels], Array(16).fill(137));
  assert.deepEqual(events, expectedEvents); assert.deepEqual(events, ['update', 'update', 'update']);
});

test('an ordinary no-data material allocation matches source callbacks and waits for a requested ready version', async t => {
  const f = fixture(t), input = new T.DataTexture(null, 3, 2), oracle = new T.DataTexture(null, 3, 2);
  t.after(() => {input.dispose(); oracle.dispose();});
  input.source.dataReady = oracle.source.dataReady = false; input.needsUpdate = oracle.needsUpdate = true;
  f.material.normalMap = input; f.material.normalMapType = T.ObjectSpaceNormalMap;
  const source = sourceTextures(t, f.device), events = [], expectedEvents = [];
  input.onUpdate = texture => {assert.equal(texture, input); events.push('update');};
  oracle.onUpdate = () => expectedEvents.push('update');
  source.update(oracle); await f.renderer.renderAsync(f.scene, f.camera);
  const initial = sampled(draws(f)[0]), expected = source.allocation(oracle);
  assert.deepEqual([initial.width, initial.height, initial.format], [expected.width, expected.height, expected.format]);
  assert.deepEqual([initial.width, initial.height], [3, 2]); assert.equal(mapUploads(f).length, 0);
  assert.deepEqual(events, expectedEvents); assert.equal(initial.pixels, undefined);
  input.image.data = oracle.image.data = new Uint8Array(24).fill(67);
  input.source.dataReady = oracle.source.dataReady = true;
  source.update(oracle); await f.renderer.renderAsync(f.scene, f.camera);
  assert.equal(sampled(draws(f).at(-1)), initial); assert.equal(mapUploads(f).length, 0); assert.deepEqual(events, expectedEvents);
  input.needsUpdate = oracle.needsUpdate = true;
  source.update(oracle); await f.renderer.renderAsync(f.scene, f.camera);
  assert.deepEqual([...sampled(draws(f).at(-1)).pixels], Array(24).fill(67));
  assert.equal(mapUploads(f).length, 1); assert.deepEqual(events, expectedEvents); assert.deepEqual(events, ['update', 'update']);
});
