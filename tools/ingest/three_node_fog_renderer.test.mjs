/** Authored scene fog through the production renderer and recording WebGPU
 * device. Verifies source callbacks, accepted uniforms and resource ownership;
 * pixel parity is a separate browser test. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const N = T.TSL;
const isFog = draw => /authoredSceneFog/.test(draw.pipeline.fragment?.module?.code ?? '');
const values = draw => [...draw.contents.values()].flatMap(bytes =>
  [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);

async function fixture(t, Material = T.MeshBasicMaterial) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => ({dimension: '2d', sampleCount: 1, ...descriptor,
    width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
    depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = (destination, data) => {
    destination.texture.pixels = new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice();
  };
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  let configuration;
  const canvas = {width: 16, height: 16, style: {}, getContext: () => ({configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [16, 16], format: configuration.format, usage: configuration.usage}); }})};
  const Renderer = createWebGPURendererClass(T), renderer = new Renderer({canvas, device});
  renderer.outputColorSpace = T.LinearSRGBColorSpace; t.after(() => renderer.dispose()); await renderer.init();
  const scene = new T.Scene(), material = new Material(), object = new T.Mesh(new T.PlaneGeometry(), material);
  object.frustumCulled = false; scene.add(object, new T.AmbientLight(0xffffff, .5));
  scene.fog = new T.Fog(new T.Color(.875, .9375, .8125), 1, 8);
  scene.userData.fogColor = new T.Color(.1875, .3125, .4375);
  const updates = [];
  const color = N.uniform(new T.Color()).setName('authoredSceneFog').onObjectUpdate(context => {
    updates.push({scene: context.scene, object: context.object, material: context.material, camera: context.camera});
    return context.object.userData.fogColor ?? context.scene.userData.fogColor;
  });
  scene.fogNode = N.fog(color, N.float(.5));
  const camera = new T.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  t.after(() => { object.geometry.dispose(); material.dispose(); });
  return {renderer, scene, material, object, camera, device, updates, color};
}

function sourceObserver(f) {
  const builder = new T.WGSLNodeBuilder(f.object, f.renderer);
  builder.scene = f.scene; builder.camera = f.camera; builder.fogNode = f.scene.fogNode;
  builder.lightsNode = N.lights([]); builder.build();
  const renderObject = {object: f.object, geometry: f.object.geometry, material: f.material,
    scene: f.scene, camera: f.camera, context: {width: 16, height: 16}, bundle: null,
    lightsNode: builder.lightsNode};
  return () => builder.observer.needsRefresh(renderObject, {renderer: f.renderer, renderId: f.renderer.info.calls + 1});
}

for (const Material of [T.MeshBasicMaterial, T.MeshPhongMaterial, T.MeshStandardMaterial,
  T.MeshPhysicalMaterial, T.MeshBasicNodeMaterial, T.MeshToonNodeMaterial])
test(Material.name + ' automatically uses authored scene fog without authored material slots', async t => {
  const f = await fixture(t, Material);
  f.renderer.render(f.scene, f.camera);
  assert.deepEqual(f.updates, [{scene: f.scene, object: f.object, material: f.material, camera: f.camera}]);
  await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat().filter(isFog); assert.equal(draws.length, 1);
  assert.ok(values(draws[0]).includes(.1875)); assert.ok(!values(draws[0]).includes(.875));
  assert.equal(f.material.colorNode == null, true); assert.equal(f.updates.length, 1);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('queued fog values, graph replacement and removal preserve every accepted render', async t => {
  const f = await fixture(t);
  f.renderer.render(f.scene, f.camera);
  f.scene.userData.fogColor.setRGB(.6875, .8125, .9375); f.object.position.x = .125;
  f.renderer.render(f.scene, f.camera);
  f.scene.fogNode = N.fog(N.uniform(new T.Color(.5625, .625, .75)).setName('authoredSceneFog'), N.float(.25));
  f.renderer.render(f.scene, f.camera);
  f.scene.fogNode = null; f.renderer.render(f.scene, f.camera);
  f.scene.userData.fogColor.setRGB(99, 99, 99);
  assert.equal(f.updates.length, 2);
  await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat(), native = draws.filter(isFog);
  assert.equal(native.length, 3);
  for (const [index, value] of [.1875, .6875, .5625].entries()) assert.ok(values(native[index]).includes(value));
  assert.ok(!values(native[0]).includes(.6875)); assert.ok(!values(native[1]).includes(.1875));
  assert.equal(draws.filter(draw => draw.pipeline.vertex.entryPoint === 'vertex_main').length, 1);
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.renderer._materialCaptureValues, 0);
});

test('ordinary fog follows the source observer SHARED policy until an object changes', async t => {
  const f = await fixture(t), observe = sourceObserver(f), {FULL, SHARED} = T.RenderObjectRefreshType;
  assert.equal(observe(), FULL); f.renderer.render(f.scene, f.camera);
  f.scene.userData.fogColor.setRGB(.6875, .8125, .9375);
  assert.equal(observe(), SHARED); f.renderer.render(f.scene, f.camera);
  f.camera.position.x = .25;
  assert.equal(observe(), SHARED); f.renderer.render(f.scene, f.camera);
  f.object.position.x = .125; f.scene.updateMatrixWorld();
  assert.equal(observe(), FULL); f.renderer.render(f.scene, f.camera);
  assert.equal(f.updates.length, 4, 'source callbacks run even when their object-group upload is retained');
  await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat().filter(isFog); assert.equal(draws.length, 4);
  for (const [index, value] of [.1875, .1875, .1875, .6875].entries()) assert.ok(values(draws[index]).includes(value));
  assert.ok(!values(draws[1]).includes(.6875)); assert.ok(!values(draws[2]).includes(.6875));
  assert.notDeepEqual(values(draws[0]), values(draws[2]), 'camera shared uniforms still update');
});

test('render-group fog inputs stay live on ordinary materials with source SHARED refresh', async t => {
  const f = await fixture(t); f.color.setGroup(N.renderGroup);
  const observe = sourceObserver(f), {FULL, SHARED} = T.RenderObjectRefreshType;
  assert.equal(observe(), FULL); f.renderer.render(f.scene, f.camera);
  f.scene.userData.fogColor.setRGB(.6875, .8125, .9375);
  assert.equal(observe(), SHARED); f.renderer.render(f.scene, f.camera);
  await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat().filter(isFog);
  assert.equal(draws.length, 2); assert.ok(values(draws[0]).includes(.1875)); assert.ok(values(draws[1]).includes(.6875));
  assert.equal(f.updates.length, 2);
});

test('authored material slots keep scene fog object uniforms live under the source FULL policy', async t => {
  const f = await fixture(t); f.material.colorNode = N.materialColor;
  const observe = sourceObserver(f);
  assert.equal(observe(), T.RenderObjectRefreshType.FULL); f.renderer.render(f.scene, f.camera);
  f.scene.userData.fogColor.setRGB(.6875, .8125, .9375);
  assert.equal(observe(), T.RenderObjectRefreshType.FULL); f.renderer.render(f.scene, f.camera);
  await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat().filter(isFog);
  assert.equal(draws.length, 2); assert.ok(values(draws[0]).includes(.1875)); assert.ok(values(draws[1]).includes(.6875));
});

test('authored fog shadows unused ordinary parameters and leaves fog-disabled companions alone', async t => {
  const f = await fixture(t);
  // The unused ordinary graph has no range to interpolate; the authored graph
  // is the source-selected output and must not acquire fixed-fog validation.
  f.scene.fog.near = f.scene.fog.far;
  const other = new T.Mesh(f.object.geometry, new T.MeshBasicMaterial({fog: false}));
  other.frustumCulled = false; f.scene.add(other); t.after(() => other.material.dispose());
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(f.updates.length, 1);
  const draws = f.device.snapshots.flat(); assert.equal(draws.filter(isFog).length, 1);
  assert.equal(draws.filter(draw => draw.pipeline.vertex.entryPoint === 'vertex_main').length, 1);
});

test('fog-disabled scenes retain fixed ArrayCamera rendering without evaluating an authored graph', async t => {
  const f = await fixture(t); f.material.fog = false;
  const camera = new T.ArrayCamera([f.camera]);
  f.renderer.render(f.scene, camera); await f.renderer.waitForGPU();
  assert.equal(f.updates.length, 0); assert.equal(f.device.snapshots.flat().filter(isFog).length, 0);
  assert.equal(f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'vertex_main').length, 1);
});

test('compileAsync prepares authored fog without invoking its draw updates', async t => {
  const f = await fixture(t);
  await f.renderer.compileAsync(f.scene, f.camera); assert.equal(f.updates.length, 0);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  assert.equal(f.updates.length, 1); assert.equal(f.device.snapshots.flat().filter(isFog).length, 1);
});

test('authored fog shares native shadow ownership while shadow passes keep fog inactive', async t => {
  const f = await fixture(t, T.MeshStandardMaterial);
  f.renderer.shadowMap.enabled = true; f.renderer.shadowMap.type = T.BasicShadowMap; f.object.receiveShadow = true;
  const light = new T.DirectionalLight(); light.position.set(1, 2, 3); light.castShadow = true;
  light.shadow.mapSize.set(16, 16); f.scene.add(light);
  const caster = new T.Mesh(new T.BoxGeometry(.5, .5, .5), new T.MeshBasicMaterial({fog: false}));
  caster.castShadow = true; caster.frustumCulled = false; f.scene.add(caster);
  t.after(() => { caster.geometry.dispose(); caster.material.dispose(); });
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const diagnostics = f.renderer._dispatcher.entry(f.scene).bridge.diagnostics;
  assert.equal(diagnostics.shadowBytes, 0); assert.equal(diagnostics.shadowStats, null);
  assert.equal(f.device.snapshots.flat().filter(isFog).length, 1); assert.equal(f.updates.length, 1);
  assert.ok(f.device.passes.some(pass => pass.desc.depthStencilAttachment?.view?.texture?.width === 16 && pass.draws.length));
});

test('authored fog textures retain source SHARED bindings and replace them on FULL refresh', async t => {
  const f = await fixture(t), textures = [17, 91].map(value => {
    const texture = new T.DataTexture(new Uint8Array(16).fill(value), 2, 2);
    texture.needsUpdate = true; t.after(() => texture.dispose()); return texture;
  });
  const input = N.texture(textures[0]); f.scene.fogNode = N.fog(input.rgb, N.float(.5));
  const observe = sourceObserver(f);
  assert.equal(observe(), T.RenderObjectRefreshType.FULL);
  f.renderer.render(f.scene, f.camera); input.value = textures[1]; f.renderer.render(f.scene, f.camera);
  f.object.position.x = .125; f.scene.updateMatrixWorld();
  assert.equal(observe(), T.RenderObjectRefreshType.FULL); f.renderer.render(f.scene, f.camera);
  await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
  const sampled = draw => [...draw.groups.values()].flatMap(({group}) => group.entries ?? []).map(entry => entry.resource?.texture).filter(Boolean);
  assert.equal(draws.length, 3);
  assert.ok(sampled(draws[0]).some(texture => texture.pixels?.[0] === 17));
  assert.ok(sampled(draws[1]).some(texture => texture.pixels?.[0] === 17));
  assert.ok(sampled(draws[2]).some(texture => texture.pixels?.[0] === 91));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const order of [-1, 1]) test(`SHARED fog sees another FULL draw's uploaded pixels with renderOrder=${order}`, async t => {
  const f = await fixture(t), pixels = new Uint8Array(16).fill(17), texture = new T.DataTexture(pixels, 2, 2);
  texture.needsUpdate = true; t.after(() => texture.dispose());
  f.scene.fogNode = N.fog(N.texture(texture).rgb, N.float(.5)); f.object.renderOrder = order;
  const material = new T.MeshBasicNodeMaterial(); material.colorNode = N.materialColor;
  const other = new T.Mesh(f.object.geometry, material); other.frustumCulled = false; f.scene.add(other);
  t.after(() => material.dispose());
  f.renderer.render(f.scene, f.camera);
  pixels.fill(91); texture.needsUpdate = true; f.renderer.render(f.scene, f.camera);
  other.visible = false; pixels.fill(177); texture.needsUpdate = true; f.renderer.render(f.scene, f.camera);
  other.visible = true; pixels.fill(220); texture.needsUpdate = true; f.renderer.render(f.scene, f.camera);
  pixels.fill(255); texture.needsUpdate = true;
  await f.renderer.waitForGPU();
  const draws = f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
  const sampled = draw => [...draw.groups.values()].flatMap(({group}) => group.entries ?? [])
    .map(entry => entry.resource?.texture).filter(Boolean);
  assert.equal(draws.length, 7);
  for (const [index, value] of [17, 17, 91, 91, 91, 220, 220].entries())
    assert.ok(sampled(draws[index]).some(input => input.pixels?.[0] === value), `draw ${index} retains source pixel ${value}`);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});
