/** Decoded native cubes use existing immutable image snapshots and texture
 * uploads. Pinned upload calls and WGSL are the oracles; host classes and GPU
 * commands are recorded doubles, not browser decode or pixel-parity evidence. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {createGpuThreeTextures, inspectThreeTexture} from './three_textures.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const N = THREE.TSL, Renderer = createWebGPURendererClass(THREE, {maxDraws: 64, maxBindings: 64});
function platform(t) {
  const keys = ['ImageData', 'ImageBitmap', 'HTMLCanvasElement', 'OffscreenCanvas', 'HTMLImageElement', 'document', 'structuredClone', 'createImageBitmap'];
  const saved = new Map(keys.map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
  const clones = [], bitmapCaptures = [];
  class ImageData {
    constructor(data = [1, 2, 3, 255], width = 1, height = 1) {
      this.data = new Uint8ClampedArray(data); this.width = width; this.height = height; this.colorSpace = 'srgb';
    }
  }
  class ImageBitmap {
    constructor(data = [1, 2, 3, 255], width = 1, height = 1) {
      this.pixels = [...data]; this.width = width; this.height = height; this.closeCalls = 0; this.originClean = true;
    }
    close() { this.closeCalls++; this.width = 0; this.height = 0; this.pixels = null; }
  }
  class HTMLCanvasElement {
    constructor(data = [1, 2, 3, 255], width = 1, height = 1) {
      this.width = width; this.height = height; this.pixels = [...data]; this.originClean = true;
    }
    getContext() { assert.fail('Capture must not request a canvas context or rasterize caller pixels'); }
  }
  class OffscreenCanvas extends HTMLCanvasElement {
    constructor(width, height) { super(Array(width * height * 4).fill(0), width, height); }
  }
  class HTMLImageElement {
    constructor(data = [1, 2, 3, 255], width = 1, height = 1) {
      this.width = this.naturalWidth = width; this.height = this.naturalHeight = height;
      this.pixels = [...data]; this.complete = true; this.originClean = true;
    }
  }
  Object.assign(globalThis, {ImageData, ImageBitmap, HTMLCanvasElement, OffscreenCanvas, HTMLImageElement,
    document: {createElement() { assert.fail('Capture must not create an intermediate DOM canvas'); }},
    createImageBitmap(image, options) {
      assert.deepEqual(options, {premultiplyAlpha: image instanceof HTMLImageElement ? 'none' : 'default', colorSpaceConversion: 'default'});
      if (image.captureError) throw image.captureError;
      if (!image.originClean) return Promise.reject(new DOMException('tainted image capture', 'SecurityError'));
      const copy = new ImageBitmap(image.pixels, image.naturalWidth ?? image.width, image.naturalHeight ?? image.height);
      copy.originClean = image.originClean; bitmapCaptures.push(copy);
      return Promise.resolve(copy);
    },
    structuredClone(image, options) {
      assert.equal(options, undefined, 'snapshot must not transfer the application image');
      if (image instanceof ImageBitmap && !image.originClean) throw new DOMException('tainted', 'DataCloneError');
      const copy = image instanceof ImageBitmap
        ? new ImageBitmap(image.pixels, image.width, image.height)
        : new ImageData(image.data, image.width, image.height);
      clones.push(copy); return copy;
    }});
  t.after(() => {
    for (const [key, descriptor] of saved) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
    }
  });
  return {ImageData, ImageBitmap, HTMLCanvasElement, OffscreenCanvas, HTMLImageElement, clones, bitmapCaptures};
}
function fixture(t, Material = THREE.MeshStandardNodeMaterial) {
  const device = geometryDevice(); device.textures = []; device.textureWrites = []; device.imageCopies = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = (destination, data) => device.textureWrites.push({
    texture: destination.texture, level: destination.mipLevel ?? 0, layer: destination.origin?.[2] ?? 0,
    bytes: new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice()});
  device.queue.copyExternalImageToTexture = (source, destination, extent) => device.imageCopies.push({
    texture: destination.texture, image: source.source, pixels: [...(source.source.data ?? source.source.pixels)],
    level: destination.mipLevel ?? 0, layer: destination.origin?.[2] ?? destination.origin?.z ?? 0,
    flipY: source.flipY, premultipliedAlpha: destination.premultipliedAlpha, colorSpace: destination.colorSpace,
    width: extent[0] ?? extent.width, height: extent[1] ?? extent.height});
  let configuration;
  const canvas = {width: 16, height: 16, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [16, 16], format: configuration.format, usage: 16}); }
  })};
  const renderer = new Renderer({canvas, device}); t.after(() => renderer.dispose());
  const material = new Material(Material === THREE.MeshBasicNodeMaterial ? {} : {roughness: .5, metalness: .25});
  material.colorNode = N.uniform(new THREE.Color(.5, .25, .125)).setName('cubeEnvironmentSurface');
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(), material), scene = new THREE.Scene(); scene.add(mesh);
  const camera = new THREE.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  return {device, renderer, material, mesh, scene, camera};
}
function imageCube(host, kind = 'ImageBitmap', size = 2, value = 10) {
  const faces = Array.from({length: 6}, (_, i) => {
    const bytes = Array(size * size * 4).fill(value + i);
    if (kind === 'OffscreenCanvas') { const face = new host.OffscreenCanvas(size, size); face.pixels = bytes; return face; }
    return new host[kind](bytes, size, size);
  });
  const texture = new THREE.CubeTexture(faces); texture.colorSpace = THREE.SRGBColorSpace;
  texture.generateMipmaps = false; texture.minFilter = texture.magFilter = THREE.LinearFilter;
  texture.needsUpdate = true; return texture;
}
const views = draw => [...draw.groups.values()].flatMap(({group}) => group.entries)
  .map(entry => entry.resource).filter(resource => resource?.texture);
const surfaces = device => device.snapshots.flat().filter(draw => draw.pipeline.fragment?.module?.code?.includes('cubeEnvironmentSurface'));
const cubeView = draw => views(draw).find(resource => resource.dimension === 'cube');
const copies = (device, texture) => device.imageCopies.filter(copy => copy.texture === texture);
const normalize = copy => ({level: copy.level, layer: copy.layer, pixels: copy.pixels, flipY: copy.flipY,
  premultipliedAlpha: copy.premultipliedAlpha, colorSpace: copy.colorSpace, width: copy.width, height: copy.height});
const sortCopies = values => values.map(normalize).sort((a,b) => a.layer-b.layer || a.level-b.level);
function sourceCopies(texture) {
  const result = [], backend = new THREE.WebGPUBackend();
  backend.device = {queue: {copyExternalImageToTexture(source, destination, extent) {
    result.push({level: destination.mipLevel, layer: destination.origin.z,
      pixels: [...(source.source.data ?? source.source.pixels)], flipY: source.flipY,
      premultipliedAlpha: destination.premultipliedAlpha, colorSpace: destination.colorSpace,
      width: extent.width, height: extent.height});
  }}};
  backend.textureUtils._copyCubeMapToTexture(texture, {}, {format: 'rgba8unorm',
    size: {width: texture.image[0].width, height: texture.image[0].height, depthOrArrayLayers: 6}});
  return sortCopies(result);
}
function useCube(f, texture) {
  f.material.lights = false;
  f.material.colorNode = N.cubeTexture(texture, N.vec3(.25, .5, 1)).level(0).rgb
    .mul(N.uniform(1).setName('cubeEnvironmentSurface'));
}

for (const kind of ['ImageData', 'ImageBitmap', 'HTMLImageElement', 'HTMLCanvasElement', 'OffscreenCanvas'])
  test(`native ${kind} cube graphs match pinned WGSL and six source upload faces`, async t => {
    const host = platform(t), f = fixture(t, THREE.MeshBasicNodeMaterial); await f.renderer.init();
    const texture = imageCube(host, kind); useCube(f, texture);
    f.scene.updateMatrixWorld(); f.camera.coordinateSystem = THREE.WebGPUCoordinateSystem;
    f.camera.updateProjectionMatrix(); f.camera.updateMatrixWorld();
    const programs = createThreeNodePrograms({three: THREE, renderer: f.renderer}); t.after(() => programs.dispose());
    const descriptor = programs.compile(f.material, f.mesh, {scene: f.scene, camera: f.camera, lights: []});
    const builder = new THREE.WGSLNodeBuilder(f.mesh, f.renderer); builder.scene = f.scene; builder.camera = f.camera;
    builder.lightsNode = N.lights([]); builder.build();
    assert.equal(descriptor.vertex, builder.vertexShader); assert.equal(descriptor.fragment, builder.fragmentShader);
    assert.equal(host.clones.length + host.bitmapCaptures.length, 0, 'construction performs no image copy');
    const expected = sourceCopies(texture); f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
    const view = cubeView(surfaces(f.device)[0]);
    assert.equal(view.texture.depthOrArrayLayers, 6); assert.equal(view.format, 'rgba8unorm-srgb');
    assert.deepEqual(sortCopies(copies(f.device, view.texture)), expected);
    assert.equal(f.renderer._nodeCaptureBytes, 0);
  });

test('authored ImageBitmap cube mips retain source face swaps, orientation and premultiplication', async t => {
  const host = platform(t), f = fixture(t, THREE.MeshBasicNodeMaterial); await f.renderer.init();
  const texture = imageCube(host); texture.mipmaps = [imageCube(host, 'ImageBitmap', 1, 30)];
  texture.flipY = true; texture.premultiplyAlpha = true; texture.minFilter = THREE.LinearMipmapLinearFilter;
  useCube(f, texture); const expected = sourceCopies(texture);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const view = cubeView(surfaces(f.device)[0]), actual = copies(f.device, view.texture);
  assert.equal(actual.length, 12); assert.equal(view.texture.mipLevelCount, 2);
  assert.deepEqual(sortCopies(actual), expected);
  assert.equal(actual.find(copy => copy.level === 0 && copy.layer === 2).pixels[0], 13);
  assert.equal(actual.find(copy => copy.level === 0 && copy.layer === 3).pixels[0], 12);
  assert.equal(host.clones.length, 12);
});

test('queued canvas cube versions retain per-face pixels, flip and alpha before async upload', async t => {
  const host = platform(t), f = fixture(t, THREE.MeshBasicNodeMaterial); await f.renderer.init();
  const texture = imageCube(host, 'HTMLCanvasElement'); useCube(f, texture);
  const a = sourceCopies(texture); f.renderer.render(f.scene, f.camera);
  texture.image.forEach((face, i) => face.pixels.fill(30+i));
  texture.flipY = true; texture.premultiplyAlpha = true; texture.needsUpdate = true;
  const b = sourceCopies(texture); f.renderer.render(f.scene, f.camera);
  texture.image.forEach(face => face.pixels.fill(99));
  f.renderer.render(f.scene, f.camera);
  await f.renderer.waitForGPU();
  const draws = surfaces(f.device); assert.equal(draws.length, 3);
  const first = cubeView(draws[0]).texture, second = cubeView(draws[1]).texture;
  assert.notEqual(first, second); assert.equal(cubeView(draws[2]).texture, second);
  assert.deepEqual(sortCopies(copies(f.device, first)), a);
  assert.deepEqual(sortCopies(copies(f.device, second)), b);
  assert.equal(host.bitmapCaptures.length, 12, 'unmarked pixels do not replace a captured source version');
  assert.equal(texture.image[0].pixels[0], 99);
});

test('generated decoded cube mipmaps retain the full per-face sRGB pyramid', async t => {
  const host = platform(t), f = fixture(t, THREE.MeshBasicNodeMaterial); await f.renderer.init();
  const texture = imageCube(host, 'ImageBitmap', 4); texture.generateMipmaps = true;
  texture.minFilter = THREE.LinearMipmapLinearFilter; useCube(f, texture);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const native = cubeView(surfaces(f.device)[0]).texture;
  assert.equal(native.mipLevelCount, 3); assert.equal(copies(f.device, native).length, 6);
  const mipDraws = f.device.snapshots.flat().filter(draw => draw.pipeline.fragment?.module?.code?.includes('linearSampler'));
  assert.equal(mipDraws.length, 12);
  assert.equal(texture.generateMipmaps, true);
});

test('shared application bitmaps are cloned once and only private cube images are retired', async t => {
  const host = platform(t), f = fixture(t, THREE.MeshBasicNodeMaterial); await f.renderer.init();
  const bitmap = new host.ImageBitmap(Array(16).fill(5), 2, 2);
  const texture = imageCube(host); texture.image = Array(6).fill(bitmap); useCube(f, texture);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(host.clones.length, 1); assert.equal(copies(f.device, cubeView(surfaces(f.device)[0]).texture).length, 6);
  f.renderer.dispose();
  assert.equal(bitmap.closeCalls, 0); assert.equal(bitmap.width, 2);
  assert.equal(host.clones[0].closeCalls, 1);
});

for (const [Material, location] of [[THREE.MeshStandardMaterial, 'scene'], [THREE.MeshPhysicalMaterial, 'material']])
  test(`${Material.name} prepares decoded ${location} cube PMREM before ordinary PBR draws`, async t => {
    const host = platform(t), f = fixture(t, Material); await f.renderer.init();
    f.material.colorNode = null;
    const texture = imageCube(host, 'HTMLImageElement', 16);
    if (location === 'scene') f.scene.environment = texture; else f.material.envMap = texture;
    const expected = sourceCopies(texture); f.renderer.render(f.scene, f.camera);
    texture.image.forEach(face => face.pixels.fill(99)); await f.renderer.waitForGPU();
    const all = f.device.snapshots.flat(), conversions = all.filter(draw => draw.pipeline.fragment?.module?.code?.includes('texture_cube<f32>'));
    assert.equal(conversions.length, 1); assert.equal(f.renderer.info.f3d.offscreenRenders, 13);
    assert.deepEqual(sortCopies(copies(f.device, cubeView(conversions[0]).texture)), expected);
    const consumers = all.filter(draw => draw.pipeline.fragment?.module?.code?.includes('iblIrradiance')); assert.equal(consumers.length, 1);
    assert.match(consumers[0].pipeline.fragment.module.code, /roughnessToMip/);
    assert.ok(all.indexOf(conversions[0]) < all.indexOf(consumers[0]));
    assert.equal(f.renderer._nodeCaptureBytes, 0);
  });

test('decoded cube PMREM refresh preserves queued image versions before each consumer', async t => {
  const host = platform(t), f = fixture(t); await f.renderer.init();
  const texture = imageCube(host, 'HTMLCanvasElement', 16); f.scene.environment = texture;
  const a = sourceCopies(texture); f.renderer.render(f.scene, f.camera);
  texture.image.forEach((face, i) => face.pixels.fill(40+i)); texture.needsUpdate = true; texture.needsPMREMUpdate = true;
  const b = sourceCopies(texture); f.renderer.render(f.scene, f.camera);
  texture.image.forEach(face => face.pixels.fill(99)); await f.renderer.waitForGPU();
  const all = f.device.snapshots.flat(), converted = all.filter(draw => draw.pipeline.fragment?.module?.code?.includes('texture_cube<f32>'));
  const consumers = surfaces(f.device); assert.equal(converted.length, 2); assert.equal(consumers.length, 2);
  assert.deepEqual(sortCopies(copies(f.device, cubeView(converted[0]).texture)), a);
  assert.deepEqual(sortCopies(copies(f.device, cubeView(converted[1]).texture)), b);
  assert.ok(all.indexOf(converted[0]) < all.indexOf(consumers[0]));
  assert.ok(all.indexOf(consumers[0]) < all.indexOf(converted[1]));
  assert.ok(all.indexOf(converted[1]) < all.indexOf(consumers[1]));
});

for (const levels of [1, 2, 3]) test(`native CPU cube flip retains pinned layer and authored-mip behavior for ${levels} levels`, async t => {
  const size = levels === 3 ? 4 : 2;
  const cube = (extent, start) => {
    const faces = Array.from({length: 6}, (_, i) => new THREE.DataTexture(
      Uint8Array.from({length: extent*extent*4}, (_, j) => start + i*20 + Math.floor(j/(extent*4))),
      extent, extent));
    const texture = new THREE.CubeTexture(faces); texture.generateMipmaps = false;
    texture.minFilter = texture.magFilter = THREE.LinearFilter; texture.flipY = true; texture.needsUpdate = true;
    return texture;
  };
  const texture = cube(size, 1); for (let level=1; level<levels; level++) texture.mipmaps.push(cube(size>>level, level+5));
  const expected = new Map(), backend = new THREE.WebGPUBackend(), flips = [];
  const reverseRows = (data,width,height) => Uint8Array.from({length: data.length}, (_, i) => data[(height-1-Math.floor(i/(width*4)))*width*4+i%(width*4)]);
  backend.device = {queue: {writeTexture(destination, data) {
    expected.set(destination.origin.z+':'+destination.mipLevel, Uint8Array.from(data));
  }}};
  // The pinned _flipY delegates to a base-level-only pass; applying that pass's
  // row permutation records the exact helper order without executing a GPU.
  backend.textureUtils._getPassUtils = () => ({flipY(native, descriptor, layer) {
    flips.push(layer); expected.set(layer+':0', reverseRows(expected.get(layer+':0'),size,size));
  }});
  backend.textureUtils._copyCubeMapToTexture(texture, {}, {format:'rgba8unorm',size:{width:size,height:size,depthOrArrayLayers:6}});
  const f = fixture(t, THREE.MeshBasicNodeMaterial), pool = createGpuThreeTextures(f.device, {three:THREE,nativeCube:true});
  t.after(() => pool.dispose()); await pool.prepare([texture]);
  const actual = new Map(f.device.textureWrites.map(write => [write.layer+':'+write.level,write.bytes]));
  assert.deepEqual(actual, expected); assert.equal(flips.length, 6*levels);
});

test('the default texture pool preserves WebGL cube face order and bitmap flag admission', async t => {
  const host = platform(t), texture = imageCube(host); texture.flipY = true;
  assert.throws(() => inspectThreeTexture(texture, THREE), {code:'THREE_TEXTURE_SOURCE'});
  assert.doesNotThrow(() => inspectThreeTexture(texture, THREE, {nativeCube:true}));
  const canvas = imageCube(host, 'HTMLCanvasElement'); canvas.flipY = true;
  const legacy = inspectThreeTexture(canvas, THREE), native = inspectThreeTexture(canvas, THREE, {nativeCube:true});
  assert.deepEqual(legacy.uploads.map(upload=>upload.layer), [0,1,2,3,4,5]);
  assert.deepEqual(native.uploads.map(upload=>upload.layer), [0,1,3,2,4,5]);
});

test('scene cubes coexist with fixed sprites, lines, points and Basic meshes', async t => {
  const host = platform(t), f = fixture(t, THREE.MeshStandardMaterial); await f.renderer.init();
  f.material.colorNode = null; f.scene.environment = imageCube(host, 'ImageBitmap', 16);
  const geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-.5, 0, 0), new THREE.Vector3(.5, 0, 0)]);
  f.scene.add(new THREE.Sprite(new THREE.SpriteMaterial()),
    new THREE.Line(geometry, new THREE.LineBasicMaterial()),
    new THREE.Points(geometry, new THREE.PointsMaterial({size: 1})),
    new THREE.Mesh(f.mesh.geometry, new THREE.MeshBasicMaterial()));
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat();
  assert.equal(draws.filter(draw => draw.pipeline.fragment?.module?.code?.includes('iblIrradiance')).length, 1);
  assert.equal(draws.filter(draw => draw.pipeline.vertex.entryPoint === 'vertex_main' &&
    !draw.pipeline.label?.startsWith('Animation HDR output ')).length, 4);
  assert.equal(f.renderer.info.f3d.offscreenRenders, 13);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('disabled scene cubes stay inert in fixed ArrayCamera views and later reactivate', async t => {
  const host = platform(t), f = fixture(t, THREE.MeshStandardMaterial); await f.renderer.init();
  f.material.colorNode = null; f.scene.environment = imageCube(host, 'HTMLCanvasElement', 16);
  f.renderer.lighting.enabled = false;
  f.renderer.render(f.scene, new THREE.ArrayCamera([f.camera])); await f.renderer.waitForGPU();
  const first = f.device.snapshots.flat().filter(draw => !draw.pipeline.label?.startsWith('Animation HDR output '));
  assert.equal(first.length, 1); assert.equal(first[0].pipeline.vertex.entryPoint, 'vertex_main');
  assert.equal(f.device.imageCopies.length, 0); assert.equal(f.renderer.info.f3d.offscreenRenders, 0);
  f.renderer.lighting.enabled = true; f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(f.device.imageCopies.length, 6); assert.equal(f.renderer.info.f3d.offscreenRenders, 13);
  assert.equal(f.device.snapshots.flat().filter(draw => draw.pipeline.fragment?.module?.code?.includes('iblIrradiance')).length, 1);
});
