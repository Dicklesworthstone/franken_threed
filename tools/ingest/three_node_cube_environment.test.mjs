/** Native cube environments use the pinned PMREM generator and F3D command
 * owner. Recorded commands establish input versions and ordering, not pixels. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = THREE.TSL, Renderer = createWebGPURendererClass(THREE, {maxDraws: 64, maxBindings: 64});
function cube(size = 16, value = 0x3400) {
  const faces = Array.from({length: 6}, (_, index) => new THREE.DataTexture(
    new Uint16Array(size * size * 4).fill(value + index), size, size, THREE.RGBAFormat, THREE.HalfFloatType));
  const texture = new THREE.CubeTexture(faces);
  texture.type = THREE.HalfFloatType; texture.colorSpace = THREE.LinearSRGBColorSpace;
  texture.generateMipmaps = false; texture.minFilter = texture.magFilter = THREE.LinearFilter;
  texture.needsUpdate = true;
  return texture;
}
function fixture(t, Material = THREE.MeshStandardNodeMaterial) {
  const device = geometryDevice(); device.textures = []; device.textureWrites = [];
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
    texture: destination.texture, layer: destination.origin?.[2] ?? 0,
    bytes: new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice()});
  let configuration;
  const canvas = {width: 16, height: 16, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [16, 16], format: configuration.format, usage: 16}); }
  })};
  const renderer = new Renderer({canvas, device}); t.after(() => renderer.dispose());
  const material = new Material({roughness: .5, metalness: .25});
  material.colorNode = N.uniform(new THREE.Color(.5, .25, .125)).setName('cubeEnvironmentSurface');
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(), material), scene = new THREE.Scene(); scene.add(mesh);
  const camera = new THREE.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  return {device, renderer, material, mesh, scene, camera};
}
const draws = device => device.snapshots.flat();
const surfaces = device => draws(device).filter(draw => draw.pipeline.fragment?.module?.code?.includes('cubeEnvironmentSurface'));
const conversions = device => draws(device).filter(draw => draw.pipeline.fragment?.module?.code?.includes('texture_cube<f32>'));
const views = draw => [...draw.groups.values()].flatMap(({group}) => group.entries)
  .map(entry => entry.resource).filter(resource => resource?.texture);
const atlas = draw => views(draw).find(view => view.texture.format === 'rgba16float' && view.texture.width >= 336)?.texture;
function assertCubeUpload(device, draw, size, value) {
  const view = views(draw).find(resource => resource.dimension === 'cube');
  assert.ok(view); assert.equal(view.texture.width, size); assert.equal(view.texture.depthOrArrayLayers, 6);
  const writes = device.textureWrites.filter(write => write.texture === view.texture);
  assert.equal(writes.length, 6);
  for (let layer = 0; layer < 6; layer++) {
    const write = writes.find(write => write.layer === layer);
    assert.ok(write); assert.equal(write.bytes.byteLength, size * size * 8);
    assert.ok([...new Uint16Array(write.bytes.buffer)].every(component => component === value + layer));
  }
}

for (const [Material, location, mapping] of [
  [THREE.MeshStandardNodeMaterial, 'scene', THREE.CubeReflectionMapping],
  [THREE.MeshPhysicalNodeMaterial, 'material', THREE.CubeRefractionMapping],
]) test(`${Material.name} automatically filters a ${location} cube environment through F3D`, async t => {
  const f = fixture(t, Material); await f.renderer.init();
  const source = cube(); source.mapping = mapping;
  if (location === 'material') {
    f.material.envMap = source;
    f.scene.environment = cube(32, 0x3800);
    f.material.clearcoat = .5; f.material.sheen = .25;
  } else f.scene.environment = source;
  const second = new THREE.Mesh(f.mesh.geometry, f.material.clone()); second.position.x = .25; f.scene.add(second);
  f.renderer.render(f.scene, f.camera);
  source.image.forEach(face => face.image.data.fill(0));
  await f.renderer.waitForGPU();
  assert.equal(f.renderer.info.f3d.offscreenRenders, 13, 'both materials share one source-filtered atlas');
  assert.equal(conversions(f.device).length, 1);
  assertCubeUpload(f.device, conversions(f.device)[0], 16, 0x3400);
  const consumers = surfaces(f.device); assert.equal(consumers.length, 2);
  assert.ok(consumers.every(draw => draw.pipeline.fragment.module.code.includes('fn roughnessToMip')));
  assert.ok(atlas(consumers[0])); assert.equal(atlas(consumers[0]), atlas(consumers[1]));
  assert.equal(atlas(consumers[0]).height, 64);
  assert.ok(draws(f.device).indexOf(conversions(f.device)[0]) < draws(f.device).indexOf(consumers[0]));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('cube PMREM refresh keeps each accepted face version before its consumer draw', async t => {
  const f = fixture(t); await f.renderer.init();
  const source = cube(); f.scene.environment = source;
  f.renderer.render(f.scene, f.camera);
  source.image.forEach((face, index) => face.image.data.fill(0x3800 + index));
  source.needsUpdate = true; source.needsPMREMUpdate = true;
  f.renderer.render(f.scene, f.camera);
  source.image.forEach(face => face.image.data.fill(0));
  await f.renderer.waitForGPU();
  const converted = conversions(f.device), consumers = surfaces(f.device), all = draws(f.device);
  assert.equal(converted.length, 2); assert.equal(consumers.length, 2);
  assert.equal(f.renderer.info.f3d.offscreenRenders, 26);
  assertCubeUpload(f.device, converted[0], 16, 0x3400);
  assertCubeUpload(f.device, converted[1], 16, 0x3800);
  assert.notEqual(views(converted[0]).find(view => view.dimension === 'cube').texture,
    views(converted[1]).find(view => view.dimension === 'cube').texture);
  assert.equal(atlas(consumers[0]), atlas(consumers[1]));
  assert.ok(all.indexOf(converted[0]) < all.indexOf(consumers[0]));
  assert.ok(all.indexOf(consumers[0]) < all.indexOf(converted[1]));
  assert.ok(all.indexOf(converted[1]) < all.indexOf(consumers[1]));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('replacing a cube environment with a different source size preserves earlier queued atlases', async t => {
  const f = fixture(t); await f.renderer.init();
  f.material.envMap = cube(); f.renderer.render(f.scene, f.camera);
  f.material.envMap = cube(32, 0x3800); f.renderer.render(f.scene, f.camera);
  await f.renderer.waitForGPU();
  const converted = conversions(f.device), consumers = surfaces(f.device);
  assert.equal(converted.length, 2); assert.equal(consumers.length, 2);
  assertCubeUpload(f.device, converted[0], 16, 0x3400);
  assertCubeUpload(f.device, converted[1], 32, 0x3800);
  assert.notEqual(atlas(consumers[0]), atlas(consumers[1]));
  assert.equal(atlas(consumers[0]).height, 64); assert.equal(atlas(consumers[1]).height, 128);
  assert.equal(f.renderer.info.f3d.offscreenRenders, 28);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('resizing an existing cube PMREM source refuses before callbacks or queued resource retirement', async t => {
  const f = fixture(t); await f.renderer.init(); let updates = 0;
  f.material.colorNode.onObjectUpdate(() => { updates++; return new THREE.Color(.5, .25, .125); });
  const source = cube(); f.scene.environment = source; f.renderer.render(f.scene, f.camera);
  const pending = f.renderer._pending.length;
  source.image = cube(32).image; source.needsUpdate = true; source.needsPMREMUpdate = true;
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_ENVIRONMENT_REFRESH'});
  assert.equal(updates, 1); assert.equal(f.renderer._pending.length, pending);
  await f.renderer.waitForGPU();
  assert.equal(surfaces(f.device).length, 1); assert.equal(atlas(surfaces(f.device)[0]).height, 64);
  assertCubeUpload(f.device, conversions(f.device)[0], 16, 0x3400);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});
