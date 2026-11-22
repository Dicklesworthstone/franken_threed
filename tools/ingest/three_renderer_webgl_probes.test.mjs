/** Pinned WebGL lighting semantics on F3D-owned pipelines; recording-GPU evidence. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.module.js';
import {WebGLLights} from '../../upstream/three.js/src/renderers/webgl/WebGLLights.js';
import {createWebGLRendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

async function fixture(t, Material = T.MeshStandardMaterial) {
  const device = geometryDevice(), textures = [], writes = [];
  device.features = new Set(['float32-filterable']);
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureDimension3D: 2048, maxTextureArrayLayers: 256});
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, mipLevelCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }};
    textures.push(texture); return texture;
  };
  device.createSampler = options => ({...options});
  device.queue.writeTexture = (destination, data) => writes.push({destination,
    bytes: new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice()});
  let configuration;
  const canvas = {width: 32, height: 24, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 24], format: configuration.format, usage: configuration.usage}); },
  })};
  const Renderer = createWebGLRendererClass(T), renderer = new Renderer({canvas, device});
  const material = new Material(), object = new T.Mesh(new T.PlaneGeometry(), material);
  object.frustumCulled = false;
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 4 / 3, .1, 20);
  camera.position.z = 4; scene.add(object); await renderer.init();
  t.after(() => { renderer.dispose(); object.geometry.dispose(); material.dispose(); });
  const render = async () => { renderer.render(scene, camera); await renderer.waitForGPU(); };
  return {device, textures, writes, renderer, material, object, scene, camera, render};
}

function probe(scale = 1) {
  const sh = new T.SphericalHarmonics3();
  sh.coefficients.forEach((value, i) => value.set((i + 1) / 16, (i + 2) / 32, (i + 3) / 64));
  return new T.LightProbe(sh, scale);
}

for (const Material of [T.MeshLambertMaterial, T.MeshPhongMaterial, T.MeshStandardMaterial, T.MeshPhysicalMaterial])
test(`${Material.name} renders source SH light probes through its own shader`, async t => {
  const f = await fixture(t, Material), lights = [probe(.5), probe(2)]; f.scene.add(...lights);
  await f.render();
  const reference = WebGLLights({has: () => true}, {}); reference.setup([...lights]); reference.setupView(lights, f.camera);
  const uniforms = f.renderer._programSupport.uniformsFor(f.material);
  assert.deepEqual(uniforms.lightProbe.value.map(value => value.toArray()), reference.state.probe.map(value => value.toArray()));
  assert.ok(f.device.snapshots.flat().some(draw => draw.pipeline.fragment.module.code.includes('getLightProbeIrradiance')));
  assert.equal(f.renderer.info.f3d.route, 'general-webgpu');
});

test('source probes respect ancestor visibility and camera layers and update live coefficients', async t => {
  const f = await fixture(t), visible = probe(2), hidden = probe(8), layered = probe(16), parent = new T.Group();
  parent.visible = false; parent.add(hidden); layered.layers.set(1); f.scene.add(visible, parent, layered);
  await f.render();
  let uniforms = f.renderer._programSupport.uniformsFor(f.material);
  assert.equal(uniforms.lightProbe.value[0].x, .125);
  visible.intensity = 4; visible.sh.coefficients[0].x = .25; await f.render();
  uniforms = f.renderer._programSupport.uniformsFor(f.material);
  assert.equal(uniforms.lightProbe.value[0].x, 1);
  parent.visible = true; f.camera.layers.enable(1); await f.render();
  assert.equal(f.renderer._programSupport.uniformsFor(f.material).lightProbe.value[0].x, 2.5);
});

test('probe SH is captured before source object callbacks mutate it', async t => {
  const f = await fixture(t), light = probe(2); f.scene.add(light);
  let calls = 0;
  f.object.onBeforeRender = () => { calls++; light.intensity = 8; };
  await f.render();
  assert.equal(calls, 1);
  assert.equal(f.renderer._programSupport.uniformsFor(f.material).lightProbe.value[0].x, .125);
  await f.render();
  assert.equal(calls, 2);
  assert.equal(f.renderer._programSupport.uniformsFor(f.material).lightProbe.value[0].x, .5);
});

test('adding a probe to a prepared shadow scene transitions to source program shadows', async t => {
  const f = await fixture(t), light = new T.DirectionalLight(0xffffff, 2);
  light.position.set(2, 3, 4); light.castShadow = true; light.shadow.mapSize.set(32, 16);
  f.object.castShadow = f.object.receiveShadow = true;
  f.renderer.shadowMap.enabled = true; f.renderer.shadowMap.type = T.PCFShadowMap;
  f.scene.add(light, light.target); t.after(() => light.dispose());
  await f.render();
  const before = f.device.snapshots.length;
  f.scene.add(probe(2)); await f.render();
  assert.equal(f.renderer._programSupport.uniformsFor(f.material).lightProbe.value[0].x, .125);
  const draws = f.device.snapshots.slice(before).flat();
  const receiver = draws.find(draw => draw.pipeline.fragment?.module.code.includes('getLightProbeIrradiance'));
  assert.ok(receiver, 'the receiver uses the source probe program after preparation');
  assert.ok([...receiver.groups.values()].some(({group}) => group.entries.some(entry =>
    entry.resource?.texture?.format.startsWith('depth'))), 'the probe receiver retains its shadow texture');
});
