/** Source scene environments through the actual facade and queued native draws.
 * The recording WebGPU device verifies commands and bytes, not rendered pixels. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const N = THREE.TSL;
const isEnvironment = draw => /authoredSceneEnvironment/.test(draw.pipeline.fragment?.module?.code ?? '');
const values = draw => [...draw.contents.values()].flatMap(bytes => [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);
const sampled = draw => [...draw.groups.values()].flatMap(({group}) => group.entries ?? []).map(entry => entry.resource?.texture).filter(Boolean);

async function fixture(t, Material = THREE.MeshStandardMaterial) {
  const device = geometryDevice(), uploads = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => ({dimension: '2d', sampleCount: 1, ...descriptor,
    width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
    depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = (destination, data) => uploads.push({texture: destination.texture,
    bytes: new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice()});
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  let configuration;
  const canvas = {width: 16, height: 16, style: {}, getContext: () => ({configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [16, 16], format: configuration.format, usage: configuration.usage}); }})};
  const Renderer = createWebGPURendererClass(THREE), renderer = new Renderer({canvas, device});
  t.after(() => renderer.dispose()); await renderer.init();
  const scene = new THREE.Scene(), material = new Material({roughness: .375});
  const object = new THREE.Mesh(new THREE.PlaneGeometry(), material); object.frustumCulled = false; scene.add(object);
  scene.userData.irradiance = new THREE.Color(.1875, .3125, .4375);
  const updates = [];
  scene.environmentNode = N.uniform(new THREE.Color()).setName('authoredSceneEnvironment')
    .onObjectUpdate(context => {
      updates.push({scene: context.scene, object: context.object, material: context.material, camera: context.camera});
      return context.object.userData.irradiance ?? context.scene.userData.irradiance;
    });
  const camera = new THREE.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  return {renderer, scene, material, object, camera, device, uploads, updates};
}

for (const Material of [THREE.MeshStandardMaterial, THREE.MeshPhysicalMaterial,
  THREE.MeshStandardNodeMaterial, THREE.MeshPhysicalNodeMaterial]) {
  test(`${Material.name} automatically submits authored scene irradiance through native shading`, async t => {
    const f = await fixture(t, Material);
    f.renderer.render(f.scene, f.camera);
    assert.equal(f.updates.length, 1);
    assert.deepEqual(f.updates[0], {scene: f.scene, object: f.object, material: f.material, camera: f.camera});
    await f.renderer.waitForGPU();
    const draws = f.device.snapshots.flat().filter(isEnvironment); assert.equal(draws.length, 1);
    assert.ok(values(draws[0]).includes(.1875)); assert.match(draws[0].pipeline.fragment.module.code, /iblIrradiance/);
    assert.equal(f.material.envNode == null, true); assert.equal(f.material.colorNode == null, true);
    assert.equal(f.updates.length, 1); assert.equal(f.renderer._nodeCaptureBytes, 0);
  });
}

test('queued scene environment values, graph replacement and removal keep accepted draws independent', async t => {
  const f = await fixture(t);
  f.renderer.render(f.scene, f.camera);
  f.scene.userData.irradiance = new THREE.Color(.6875, .8125, .9375); f.renderer.render(f.scene, f.camera);
  f.scene.environmentNode = N.uniform(new THREE.Color(.5625, .625, .75)).setName('authoredSceneEnvironment');
  f.renderer.render(f.scene, f.camera);
  f.scene.environmentNode = null; f.renderer.render(f.scene, f.camera);
  f.scene.userData.irradiance.setRGB(99, 99, 99);
  assert.equal(f.updates.length, 2);
  await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat(), native = draws.filter(isEnvironment);
  assert.equal(native.length, 3);
  for (const [index, value] of [.1875, .6875, .5625].entries()) assert.ok(values(native[index]).includes(value));
  assert.ok(!values(native[0]).includes(.6875)); assert.ok(!values(native[1]).includes(.1875));
  assert.equal(draws.filter(draw => draw.pipeline.vertex.entryPoint === 'vertex_main').length, 1);
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.renderer._materialCaptureValues, 0);
});

test('scene nodes override unused ordinary maps while mixed fixed objects retain their own draws', async t => {
  const f = await fixture(t);
  f.scene.environment = new THREE.Texture();
  const other = new THREE.Mesh(f.object.geometry, new THREE.MeshPhysicalMaterial());
  other.frustumCulled = false; other.userData.irradiance = new THREE.Color(.6875, .8125, .9375); f.scene.add(other);
  const geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-.5, 0, 0), new THREE.Vector3(.5, 0, 0)]);
  f.scene.add(new THREE.Mesh(f.object.geometry, new THREE.MeshBasicMaterial()),
    new THREE.Sprite(new THREE.SpriteMaterial()), new THREE.Points(geometry, new THREE.PointsMaterial()),
    new THREE.Line(geometry, new THREE.LineBasicMaterial()));
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat(), native = draws.filter(isEnvironment);
  assert.equal(native.length, 2); assert.equal(f.updates.length, 2);
  assert.ok(values(native[0]).includes(.1875)); assert.ok(values(native[1]).includes(.6875));
  assert.equal(draws.filter(draw => draw.pipeline.vertex.entryPoint === 'vertex_main').length, 4);
});

test('disabled or unused scene environments keep existing fixed ArrayCamera rendering', async t => {
  for (const inactive of ['unlit', 'disabled', 'non-node']) {
    const f = await fixture(t);
    if (inactive === 'unlit') f.object.material = new THREE.MeshBasicMaterial();
    else if (inactive === 'disabled') f.renderer.lighting.enabled = false;
    else f.scene.environmentNode = {unused: true};
    if (inactive !== 'non-node') f.scene.environment = new THREE.Texture();
    const camera = new THREE.ArrayCamera([f.camera]);
    f.renderer.render(f.scene, camera); await f.renderer.waitForGPU();
    assert.equal(f.updates.length, 0);
    assert.equal(f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'vertex_main').length, 1);
  }
});

test('compiling a scene environment runs no draw callbacks and later draws keep their source updates', async t => {
  const f = await fixture(t);
  await f.renderer.compileAsync(f.scene, f.camera); assert.equal(f.updates.length, 0);
  f.renderer.render(f.scene, f.camera);
  f.scene.userData.irradiance = new THREE.Color(.6875, .8125, .9375); f.renderer.render(f.scene, f.camera);
  await f.renderer.waitForGPU();
  assert.equal(f.updates.length, 2); assert.equal(f.device.snapshots.flat().filter(isEnvironment).length, 2);
});

test('scene-environment consumers share the native shadow owner with ordinary casters', async t => {
  const f = await fixture(t);
  f.renderer.shadowMap.enabled = true; f.renderer.shadowMap.type = THREE.BasicShadowMap; f.object.receiveShadow = true;
  const light = new THREE.DirectionalLight(); light.position.set(1, 2, 3); light.castShadow = true;
  light.shadow.mapSize.set(16, 16); f.scene.add(light);
  const caster = new THREE.Mesh(new THREE.BoxGeometry(.5, .5, .5), new THREE.MeshBasicMaterial());
  caster.castShadow = true; caster.frustumCulled = false; f.scene.add(caster);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const diagnostics = f.renderer._dispatcher.entry(f.scene).bridge.diagnostics;
  assert.equal(diagnostics.shadowBytes, 0); assert.equal(diagnostics.shadowStats, null);
  assert.equal(f.device.snapshots.flat().filter(isEnvironment).length, 1);
});

test('scene texture expressions run retained PMREM preparation through owned render targets', async t => {
  const f = await fixture(t), input = new THREE.DataTexture(new Uint16Array(64 * 32 * 4).fill(0x3800), 64, 32,
    THREE.RGBAFormat, THREE.HalfFloatType);
  input.mapping = THREE.EquirectangularReflectionMapping; input.minFilter = input.magFilter = THREE.LinearFilter;
  input.colorSpace = THREE.LinearSRGBColorSpace; input.needsUpdate = true;
  f.scene.environmentNode = N.texture(input); f.scene.environment = new THREE.Texture();
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const passes = f.renderer.info.f3d.offscreenRenders;
  assert.equal(passes, 13);
  const pbr = f.device.snapshots.flat().filter(draw => /iblIrradiance/.test(draw.pipeline.fragment?.module?.code ?? ''));
  assert.equal(pbr.length, 1);
  const atlas = sampled(pbr[0]).find(texture => texture.format === 'rgba16float' && texture.width === 336);
  assert.ok(atlas);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(f.renderer.info.f3d.offscreenRenders, passes);
  input.needsPMREMUpdate = true; f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(f.renderer.info.f3d.offscreenRenders, passes * 2);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});
