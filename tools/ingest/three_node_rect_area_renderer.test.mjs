/** Actual F3D facade, source LTC tables and recorded WebGPU bindings/uniforms.
 * No rasterized-pixel or hardware-performance claim. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {RectAreaLightTexturesLib} from './fixtures/three_rect_area_light.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const ltc = RectAreaLightTexturesLib.init();
const isArea = draw => /LTC_Evaluate/.test(draw.pipeline.fragment?.module?.code ?? '');
const values = draw => [...draw.contents.values()].flatMap(bytes => [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);
const sampled = draw => [...draw.groups.values()].flatMap(({group}) => group.entries ?? []).map(entry => entry.resource?.texture).filter(Boolean);

async function fixture(t) {
  THREE.RectAreaLightNode.setLTC(ltc);
  const device = geometryDevice(), uploads = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256});
  device.features = new Set(['float32-filterable']);
  device.createTexture = descriptor => ({dimension: '2d', sampleCount: 1, ...descriptor,
    width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
    depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = (destination, data) => uploads.push({texture: destination.texture, bytes: new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice()});
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  let configuration;
  const canvas = {width: 16, height: 16, style: {}, getContext: () => ({configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [16, 16], format: configuration.format, usage: configuration.usage}); }})};
  const Renderer = createWebGPURendererClass(THREE), renderer = new Renderer({canvas, device});
  t.after(() => renderer.dispose()); await renderer.init();
  const scene = new THREE.Scene(), material = new THREE.MeshStandardMaterial({roughness: .375});
  const object = new THREE.Mesh(new THREE.PlaneGeometry(), material); object.frustumCulled = false; scene.add(object);
  const light = new THREE.RectAreaLight(0xffffff, .5, 3.375, 1.875); light.position.set(1, 2, 3); light.color.setRGB(.375, .25, .125); scene.add(light);
  const camera = new THREE.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  return {renderer, scene, material, object, light, camera, device, uploads};
}

test('ordinary Standard area lighting submits exact source LTC bytes through the shared texture owner', async t => {
  const f = await fixture(t); f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat(); assert.equal(draws.length, 1); assert.ok(isArea(draws[0]));
  const tables = sampled(draws[0]).filter(texture => texture.width === 64 && texture.height === 64);
  assert.equal(tables.length, 2); assert.equal(f.material.colorNode, undefined);
  for (const texture of tables) {
    const upload = f.uploads.find(entry => entry.texture === texture); assert.ok(upload);
    assert.ok(Object.values(ltc).some(source => source.isDataTexture &&
      Buffer.from(source.image.data.buffer, source.image.data.byteOffset, source.image.data.byteLength).equals(Buffer.from(upload.bytes))));
  }
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('queued area-light dimensions and radiance stay independent and removal returns to fixed rendering', async t => {
  const f = await fixture(t);
  f.renderer.render(f.scene, f.camera);
  f.light.width = 6.375; f.light.height = 5.875; f.light.intensity = 1.75;
  f.renderer.render(f.scene, f.camera);
  f.scene.remove(f.light); f.renderer.render(f.scene, f.camera);
  f.light.width = 99; f.light.intensity = 99;
  await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat(); assert.equal(draws.length, 3);
  assert.ok(isArea(draws[0]) && isArea(draws[1])); assert.ok(!isArea(draws[2]));
  assert.ok(values(draws[0]).includes(1.6875)); assert.ok(!values(draws[0]).includes(3.1875));
  assert.ok(values(draws[1]).includes(3.1875)); assert.ok(!values(draws[1]).includes(1.6875));
  assert.ok(values(draws[0]).includes(.1875)); assert.ok(values(draws[1]).includes(.65625));
  assert.equal(draws[2].pipeline.vertex.entryPoint, 'vertex_main');
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.renderer._materialCaptureValues, 0);
});

test('area-light selection is captured while fixed children, sprites, lines and points keep rendering', async t => {
  const f = await fixture(t);
  const geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-.5, 0, 0), new THREE.Vector3(.5, 0, 0)]);
  f.scene.add(new THREE.Sprite(new THREE.SpriteMaterial()), new THREE.Points(geometry, new THREE.PointsMaterial()));
  f.light.add(new THREE.Line(geometry, new THREE.LineBasicMaterial()), new THREE.Mesh(f.object.geometry, new THREE.MeshBasicMaterial()));
  for (const child of f.light.children) child.frustumCulled = false;
  f.light.layers.set(1); f.renderer.render(f.scene, f.camera);
  f.light.layers.set(0); f.renderer.render(f.scene, f.camera);
  f.renderer.lighting.enabled = false; f.renderer.render(f.scene, f.camera);
  f.renderer.lighting.enabled = true; f.light.layers.set(2);
  await f.renderer.waitForGPU();
  const frames = f.device.snapshots.filter(draws => draws.length);
  assert.deepEqual(frames.map(draws => draws.length), [5, 5, 5]);
  assert.deepEqual(frames.map(draws => draws.filter(isArea).length), [0, 1, 0]);
});

test('area lights preserve source no-contribution lighting models while unlit materials stay fixed', async t => {
  const f = await fixture(t); f.scene.remove(f.object);
  for (const Material of [THREE.MeshLambertMaterial, THREE.MeshPhongMaterial, THREE.MeshToonMaterial, THREE.MeshBasicMaterial])
    f.scene.add(new THREE.Mesh(new THREE.PlaneGeometry(), new Material()));
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat(); assert.equal(draws.length, 4); assert.ok(draws.every(draw => !isArea(draw)));
  assert.equal(draws.filter(draw => draw.pipeline.vertex.entryPoint === 'vertex_main').length, 1);
});

test('inactive area lights keep ArrayCamera scenes on their existing fixed path', async t => {
  const f = await fixture(t), camera = new THREE.ArrayCamera([f.camera]);
  f.light.layers.set(1); f.renderer.render(f.scene, camera);
  f.light.layers.set(0); f.renderer.lighting.enabled = false; f.renderer.render(f.scene, camera);
  await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat(); assert.equal(draws.length, 2);
  assert.ok(draws.every(draw => draw.pipeline.vertex.entryPoint === 'vertex_main'));
});

test('area-driven lighting shares native shadow ownership without shifting fixed shadow light indices', async t => {
  for (const lit of [false, true]) {
    const f = await fixture(t); if (!lit) f.object.material = new THREE.MeshBasicMaterial();
    f.renderer.shadowMap.enabled = true; f.renderer.shadowMap.type = THREE.BasicShadowMap; f.object.receiveShadow = true;
    const directional = new THREE.DirectionalLight(); directional.position.set(1, 2, 3); directional.castShadow = true;
    directional.shadow.mapSize.set(16, 16); f.scene.add(directional);
    const caster = new THREE.Mesh(new THREE.BoxGeometry(.5, .5, .5), new THREE.MeshBasicMaterial());
    caster.castShadow = true; caster.frustumCulled = false; f.scene.add(caster);
    f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
    const diagnostics = f.renderer._dispatcher.entry(f.scene).bridge.diagnostics;
    if (lit) { assert.equal(diagnostics.shadowBytes, 0); assert.equal(diagnostics.shadowStats, null); }
    else { assert.ok(diagnostics.shadowBytes > 0); assert.equal(diagnostics.shadowStats.lightIndex, 0); }
  }
});
