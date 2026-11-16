/** Pinned source nodes and packed HDR bytes through the production renderer.
 * The GPU boundary records commands; this suite does not rasterize WGSL.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import SourceTextureUtils, {getFormat} from '../../upstream/three.js/src/renderers/webgpu/utils/WebGPUTextureUtils.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T);
const profiles = [['RGB9E5', T.UnsignedInt5999Type, 'rgb9e5ufloat'],
  ['RGB11F', T.UnsignedInt101111Type, 'rg11b10ufloat']];
function fixture(t, features = []) {
  const device = geometryDevice(), textures = [], writes = [], captures = [];
  device.features = new Set(features);
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureDimension3D: 256,
    maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture: this, ...options}; },
      destroy() { this.destroyed = true; }};
    textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = (destination, data, layout, size) => writes.push({destination,
    bytes: new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice(),
    layout: {...layout}, size: [...size]});
  let configured;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 32], format: configured.format, usage: 16}); },
  })};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  const material = new T.MeshBasicNodeMaterial(), object = new T.Mesh(new T.PlaneGeometry(), material);
  object.frustumCulled = false;
  const scene = new T.Scene(); scene.add(object);
  const camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 4;
  const capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function(request) {
    const result = capture.call(this, request); if (result) captures.push(result); return result;
  };
  t.after(() => { renderer.dispose(); material.dispose(); object.geometry.dispose(); });
  return {renderer, device, scene, camera, object, material, textures, writes, captures};
}
function texture(t, type, kind = '2d') {
  const data = new Uint32Array(2 * 2 * (kind === 'array' || kind === 'volume' ? 3 : 1) + 2).subarray(1, -1);
  data.set(Array.from({length: data.length}, (_, i) => (0xf8000000 + i * 0x10203) >>> 0));
  let value;
  if (kind === 'cube') {
    value = new T.CubeTexture(Array.from({length: 6}, (_, i) => {
      const face = new T.DataTexture(data.slice(), 2, 2, T.RGBFormat, type); face.image.data[0] += i; return face;
    }));
  } else value = kind === 'array' ? new T.DataArrayTexture(data, 2, 2, 3)
    : kind === 'volume' ? new T.Data3DTexture(data, 2, 2, 3) : new T.DataTexture(data, 2, 2);
  Object.assign(value, {type, format: T.RGBFormat, minFilter: T.LinearFilter, magFilter: T.LinearFilter,
    generateMipmaps: false, flipY: false});
  value.needsUpdate = true; t.after(() => value.dispose()); return value;
}
function sample(f, image) {
  const node = image.isCubeTexture ? N.cubeTexture(image, N.vec3(1, .5, .25))
    : image.isData3DTexture ? N.texture3D(image, N.vec3(.25, .5, .75))
      : image.isDataArrayTexture ? N.texture(image, N.vec2(.25, .75)).depth(1)
        : N.texture(image, N.vec2(.25, .75));
  f.material.colorNode = node.rgb; return node;
}
const draws = f => f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
const entries = draw => [...draw.groups.values()].flatMap(({group}) => group.entries);
const view = draw => entries(draw).find(entry => profiles.some(profile => profile[2] === entry.resource?.texture?.format)).resource;
const uploads = (f, gpu) => f.writes.filter(write => write.destination.texture === gpu);
const words = bytes => new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
function assertSource(f, descriptor) {
  const source = new T.WGSLNodeBuilder(f.object, f.renderer);
  source.scene = f.scene; source.camera = f.camera; source.build();
  const canonical = shader => shader.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
  assert.equal(canonical(descriptor.vertex), canonical(source.vertexShader));
  assert.equal(canonical(descriptor.fragment), canonical(source.fragmentShader));
}

for (const [name, type, format] of profiles) for (const kind of ['2d', 'array', 'volume', 'cube'])
test(`${name} ${kind} graphics keep pinned floating-point shaders and four-byte source texels`, async t => {
  const f = fixture(t), image = texture(t, type, kind); sample(f, image);
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  assert.equal(getFormat(image, f.device), format);
  const draw = draws(f)[0], sampled = view(draw), gpu = sampled.texture;
  assertSource(f, f.captures[0].draws[0].binding.nodeProgram);
  const dimension = kind === 'volume' ? '3d' : kind === 'array' ? '2d-array' : kind;
  const layouts = draw.pipeline.layout.bindGroupLayouts.flatMap(group => group.entries).filter(entry => entry.texture);
  assert.deepEqual(layouts.map(entry => entry.texture), [{sampleType: 'float', viewDimension: dimension, multisampled: false}]);
  assert.equal(gpu.format, format); assert.equal(gpu.usage, 1 | 2 | 4);
  assert.equal(gpu.dimension, kind === 'volume' ? '3d' : '2d');
  assert.equal(gpu.depthOrArrayLayers, kind === 'cube' ? 6 : kind === 'array' || kind === 'volume' ? 3 : 1);
  const copied = uploads(f, gpu);
  if (kind === 'cube') {
    assert.equal(copied.length, 6);
    for (let face = 0; face < 6; face++) assert.deepEqual(words(copied[face].bytes), image.image[face].image.data);
  } else {
    assert.equal(copied.length, 1); assert.deepEqual(words(copied[0].bytes), image.image.data);
    assert.deepEqual(copied[0].size, [2, 2, kind === '2d' ? 1 : 3]);
  }
  assert.ok(copied.every(write => write.layout.bytesPerRow === 8));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const [name, type] of profiles) test(`${name} queued renders retain each requested upload and ignore unrequested edits`, async t => {
  const f = fixture(t), image = texture(t, type), original = image.image, source = image.source;
  sample(f, image); const first = image.image.data.slice(), callbacks = [];
  image.onUpdate = value => callbacks.push(value);
  image.addUpdateRange(0, 4); const ranges = image.updateRanges.map(range => ({...range}));
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  image.image.data.fill(0x81234567); image.needsUpdate = true; f.renderer.render(f.scene, f.camera);
  image.image.data.fill(0xffffffff); f.renderer.render(f.scene, f.camera);
  await f.renderer.waitForGPU();
  const rendered = draws(f), gpu = rendered.map(draw => view(draw).texture);
  assert.equal(rendered.length, 3); assert.notEqual(gpu[0], gpu[1]); assert.equal(gpu[1], gpu[2]);
  assert.deepEqual(words(uploads(f, gpu[0])[0].bytes), first);
  assert.deepEqual([...words(uploads(f, gpu[1])[0].bytes)], Array(4).fill(0x81234567));
  assert.deepEqual(callbacks, [image, image]); assert.deepEqual(image.updateRanges, ranges);
  assert.equal(image.image, original); assert.equal(image.source, source);
  assert.ok(f.captures.every(frame => frame.draws[0].binding.nodeProgram === f.captures[0].draws[0].binding.nodeProgram));
});

for (const [name, type] of profiles) test(`${name} queued array layer updates compose from the preceding captured version`, async t => {
  const f = fixture(t), image = texture(t, type, 'array'); sample(f, image);
  const first = image.image.data.slice();
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  image.image.data.fill(0x9abcdef0); image.addLayerUpdate(1); image.needsUpdate = true;
  f.renderer.render(f.scene, f.camera); image.image.data.fill(0xffffffff);
  await f.renderer.waitForGPU();
  const rendered = draws(f), before = view(rendered[0]).texture, after = view(rendered[1]).texture;
  assert.deepEqual(words(uploads(f, before)[0].bytes), first);
  const expected = first.slice(); expected.fill(0x9abcdef0, 4, 8);
  assert.deepEqual(words(uploads(f, after)[0].bytes), expected);
  assert.equal(image.layerUpdates.size, 0);
});

for (const [name, type, format] of profiles) test(`${name} native source uploader has the same packed full-image and range behavior`, async t => {
  const f = fixture(t), image = texture(t, type), sourceWrites = [];
  image.addUpdateRange(0, 4); sample(f, image);
  const sourceDevice = {queue: {writeTexture(_destination, data, layout, size) {
    sourceWrites.push({bytes: new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice(),
      bytesPerRow: layout.bytesPerRow, width: size.width, height: size.height});
  }}};
  const utils = new SourceTextureUtils({device: sourceDevice, get: () => ({texture: {}, textureDescriptorGPU: {format}})});
  utils.updateTexture(image, {image: image.image});
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  image.image.data.fill(0x84020100); image.needsUpdate = true;
  utils.updateTexture(image, {image: image.image}); await f.renderer.renderAsync(f.scene, f.camera);
  const native = draws(f).map(draw => uploads(f, view(draw).texture)[0]);
  assert.equal(sourceWrites.length, 2);
  for (let i = 0; i < 2; i++) {
    assert.deepEqual(native[i].bytes, sourceWrites[i].bytes);
    assert.equal(native[i].layout.bytesPerRow, sourceWrites[i].bytesPerRow);
    assert.deepEqual(native[i].size, [sourceWrites[i].width, sourceWrites[i].height, 1]);
  }
  assert.deepEqual(image.updateRanges, [{start: 0, count: 4}]);
});

for (const [name, type] of profiles) test(`${name} nonrenderable copy destinations refuse mip generation before capturing or allocating`, async t => {
  const f = fixture(t), destination = texture(t, type), source = texture(t, type);
  destination.mipmaps = [destination.image, {width: 1, height: 1, data: new Uint32Array([0x81234567])}];
  destination.minFilter = T.LinearMipmapLinearFilter; sample(f, destination);
  await f.renderer.init(); await f.renderer.renderAsync(f.scene, f.camera);
  const allocated = f.textures.length, written = f.writes.length;
  const inputs = f.renderer._getSampledTextureInputs(), captured = {...inputs.diagnostics};
  // Native copy generation follows this live property even if the accepted
  // upload and its authored storage have not been invalidated by needsUpdate.
  destination.generateMipmaps = true;
  assert.throws(() => f.renderer.copyTextureToTexture(source, destination), {code: 'THREE_COMPUTE_TEXTURE_COPY'});
  assert.equal(f.textures.length, allocated); assert.equal(f.writes.length, written);
  assert.deepEqual(inputs.diagnostics, captured);
});
