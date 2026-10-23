/** Public PMREM retains r186 construction and filtering while F3D owns every
 * submission. The recording GPU checks resources and ordering, not pixels. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {RoomEnvironment} from '../../upstream/three.js/examples/jsm/environments/RoomEnvironment.js';
import {createPMREMGeneratorClass, createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {nativeTSLScenarios} from '../../tests/e2e/webgpu_renderer/scenarios_native_tsl.js';
import {threeBackgroundFrame} from './three_background.mjs';
import {packAnimationBackgroundFrame} from './animation_background.mjs';

const Renderer = createWebGPURendererClass(THREE, {maxDraws: 64, maxBindings: 64});
const PMREMGenerator = createPMREMGeneratorClass(THREE), N = THREE.TSL;
function panorama(width = 64) {
  const texture = new THREE.DataTexture(new Uint16Array(width * width * 2).fill(0x3800), width, width / 2,
    THREE.RGBAFormat, THREE.HalfFloatType);
  texture.mapping = THREE.EquirectangularReflectionMapping;
  texture.minFilter = texture.magFilter = THREE.LinearFilter;
  texture.colorSpace = THREE.LinearSRGBColorSpace; texture.needsUpdate = true;
  return texture;
}
function fixture(t) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.textures = [];
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      createView(options = {}) { return {texture, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  device.createSampler = descriptor => ({...descriptor});
  device.queue.writeTexture = (destination, data) => {
    destination.texture.pixels = new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice();
  };
  let configuration;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 32], format: configuration.format, usage: 16}); }
  })};
  const renderer = new Renderer({canvas, device}); t.after(() => renderer.dispose());
  return {renderer, device};
}
function observeDisposal(resources) {
  const counts = new Map(resources.map(resource => [resource, 0]));
  for (const resource of counts.keys()) resource.addEventListener('dispose', () => counts.set(resource, counts.get(resource) + 1));
  return expected => { for (const count of counts.values()) assert.equal(count, expected); };
}
function filteringResources(generator) {
  return [generator._pingPongRenderTarget, generator._blurMaterial, generator._ggxMaterial,
    ...generator._lodMeshes.map(mesh => mesh.geometry)];
}
function packet(draw) {
  const {group, offsets} = draw.groups.get(0), resource = group.entries[0].resource;
  const bytes = draw.contents.get(resource.buffer), offset = (resource.offset ?? 0) + (offsets[0] ?? 0);
  return new Float32Array(bytes.buffer, bytes.byteOffset + offset, resource.size / 4);
}
const fixedDraws = device => device.snapshots.flat().filter(draw =>
  draw.pipeline.vertex.entryPoint === 'vertex_main' && draw.streams.has(0));
function consume(renderer, texture) {
  const material = new THREE.NodeMaterial(); material.lights = false;
  material.fragmentNode = N.vec4(N.pmremTexture(texture, N.vec3(0, 0, 1), N.float(.4)))
    .mul(N.uniform(1).setName('publicPMREMResult'));
  const scene = new THREE.Scene(); scene.add(new THREE.Mesh(new THREE.PlaneGeometry(), material));
  const camera = new THREE.PerspectiveCamera(50, 1, .1, 10); camera.position.z = 3;
  renderer.setRenderTarget(null); renderer.render(scene, camera);
}

test('native public PMREM keeps retained source identity and requires initialization', t => {
  const {renderer} = fixture(t), generator = new PMREMGenerator(renderer);
  assert.ok(generator instanceof THREE.PMREMGenerator);
  assert.equal(Object.getPrototypeOf(generator), THREE.PMREMGenerator.prototype);
  assert.equal(generator.fromScene, THREE.PMREMGenerator.prototype.fromScene);
  assert.throws(() => generator.fromScene(new THREE.Scene()), /before the backend is initialized/);
});

test('public native PMREM remains sampleable after immediate generator disposal', async t => {
  const {renderer, device} = fixture(t); await renderer.init();
  const generator = new PMREMGenerator(renderer), output = generator.fromEquirectangular(panorama());
  t.after(() => output.dispose());
  assert.equal(output.isRenderTarget, true); assert.equal(output.texture.isPMREMTexture, true);
  assert.equal(output.texture.mapping, THREE.CubeUVReflectionMapping);
  assert.equal(output.texture.isF3DPMREMSource, undefined); assert.equal(output.width, 336); assert.equal(output.height, 64);
  const privateDisposal = observeDisposal([...filteringResources(generator), generator._equirectMaterial]);
  const outputDisposal = observeDisposal([output, output.texture]);
  assert.equal(generator.dispose(), undefined); privateDisposal(0); outputDisposal(0);
  await renderer.waitForGPU(); privateDisposal(1); outputDisposal(0);
  assert.equal(renderer.info.f3d.offscreenRenders, 13);
  consume(renderer, output.texture); await renderer.waitForGPU();
  const sampled = device.snapshots.flat().filter(draw => draw.pipeline.fragment?.module?.code?.includes('publicPMREMResult'));
  assert.equal(sampled.length, 1);
  const inputs = [...sampled[0].groups.values()].flatMap(({group}) => group.entries)
    .map(entry => entry.resource?.texture).filter(texture => texture?.format === 'rgba16float');
  assert.equal(inputs.length, 1); assert.equal(inputs[0].destroyed, undefined);
  assert.equal(renderer.info.f3d.offscreenRenders, 13, 'sampling a public CubeUV atlas does not bake it again');
  assert.equal(renderer._nodeCaptureBytes, 0); outputDisposal(0);
});

test('queued public PMREM resizes retire only the previous private generation', async t => {
  const {renderer} = fixture(t); await renderer.init();
  const generator = new PMREMGenerator(renderer), first = generator.fromEquirectangular(panorama());
  const firstPrivate = filteringResources(generator), firstDisposal = observeDisposal(firstPrivate);
  const conversionDisposal = observeDisposal([generator._equirectMaterial]);
  const second = generator.fromEquirectangular(panorama(128));
  t.after(() => { first.dispose(); second.dispose(); });
  const secondPrivate = filteringResources(generator), secondDisposal = observeDisposal(secondPrivate);
  const outputDisposal = observeDisposal([first, first.texture, second, second.texture]);
  assert.equal(first.height, 64); assert.equal(second.height, 128);
  assert.notEqual(first.texture, second.texture);
  assert.ok(firstPrivate.every(resource => !secondPrivate.includes(resource)));
  firstDisposal(0); secondDisposal(0); conversionDisposal(0);
  generator.dispose(); firstDisposal(0); secondDisposal(0); conversionDisposal(0);
  await renderer.waitForGPU();
  firstDisposal(1); secondDisposal(1); conversionDisposal(1); outputDisposal(0);
  assert.equal(renderer.info.f3d.offscreenRenders, 28);
  consume(renderer, first.texture); consume(renderer, second.texture); await renderer.waitForGPU();
  assert.equal(renderer.info.f3d.presentedRenders, 2); assert.equal(renderer._nodeCaptureBytes, 0);
});

test('native public fromScene renders RoomEnvironment, sigma blur and GGX through F3D', async t => {
  const {renderer} = fixture(t); await renderer.init();
  const generator = new PMREMGenerator(renderer), room = new RoomEnvironment();
  room.background = new THREE.Color(.125, .25, .5); const background = room.background;
  const previous = new THREE.RenderTarget(8, 8); renderer.setRenderTarget(previous);
  t.after(() => { room.dispose(); previous.dispose(); });
  const calls = [], render = renderer.render.bind(renderer);
  renderer.render = (root, camera) => { calls.push({root, material: root.material?.name}); return render(root, camera); };
  const output = generator.fromScene(room, .04, .2, 50, {size: 16, position: new THREE.Vector3(1, 2, 3)});
  t.after(() => output.dispose());
  assert.equal(output.depthBuffer, true); assert.equal(output.texture.isPMREMTexture, true);
  assert.equal(output.texture.f3dCapture, undefined); assert.equal(renderer.getRenderTarget(), previous);
  assert.equal(room.background, background); assert.equal(renderer.autoClear, true);
  assert.equal(calls.filter(call => call.root === room).length, 6);
  assert.equal(calls.filter(call => call.material === 'PMREM_blur').length, 2);
  assert.equal(calls.filter(call => call.material === 'PMREM_ggx').length, 12);
  const privateDisposal = observeDisposal([...filteringResources(generator),
    generator._backgroundBox.geometry, generator._backgroundBox.material]);
  generator.dispose(); privateDisposal(0); await renderer.waitForGPU(); privateDisposal(1);
  assert.equal(renderer.info.f3d.offscreenRenders, 22);
  consume(renderer, output.texture); await renderer.waitForGPU();
  assert.equal(renderer.info.f3d.presentedRenders, 1); assert.equal(renderer._nodeCaptureBytes, 0);
});

test('fromScene submits six captured face matrices and preserves its temporary null background', async t => {
  const {renderer, device} = fixture(t); await renderer.init();
  const generator = new PMREMGenerator(renderer), scene = new THREE.Scene();
  scene.background = new THREE.Color(.125, .25, .5);
  const material = new THREE.MeshBasicMaterial({color: new THREE.Color(.25, .5, .75)});
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(), material); mesh.frustumCulled = false; scene.add(mesh);
  const output = generator.fromScene(scene, 0, .1, 100, {size: 16, position: new THREE.Vector3(1, 2, 3)});
  t.after(() => output.dispose());
  const accepted = renderer._pending.filter(request => request.scene === scene);
  assert.equal(accepted.length, 6);
  assert.ok(accepted.every(request => request.backgroundState.background === null && request.frame.loadOp === 'load'));
  const expected = accepted.map(({camera}) => new Float32Array(new THREE.Matrix4()
    .multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).multiply(mesh.matrixWorld).elements));
  assert.equal(new Set(expected.map(values => Array.from(values).join(','))).size, 6);
  scene.background.setRGB(1, 0, 0); generator.dispose(); await renderer.waitForGPU();
  const draws = fixedDraws(device), faceDraws = draws.filter(draw => packet(draw)[16] === .25);
  assert.equal(faceDraws.length, 6);
  for (let i = 0; i < 6; i++) assert.deepEqual(packet(faceDraws[i]).slice(0, 16), expected[i]);
  const fixedPasses = device.passes.filter(pass => pass.draws.some(draw => draw.streams.has(0) && draw.pipeline.vertex.entryPoint === 'vertex_main'));
  assert.equal(fixedPasses.length, 7);
  assert.ok(fixedPasses.every(pass => pass.desc.colorAttachments[0].loadOp === 'load'),
    'restoring the source Color must not clear over earlier cube faces');
});

test('repeated fromScene captures keep private background colors and retire both boxes safely', async t => {
  const {renderer, device} = fixture(t); await renderer.init();
  const generator = new PMREMGenerator(renderer), scene = new THREE.Scene();
  scene.background = new THREE.Color(.125, .25, .5);
  const first = generator.fromScene(scene, 0, .1, 100, {size: 16}), firstBox = generator._backgroundBox;
  const firstDisposal = observeDisposal([firstBox.geometry, firstBox.material]);
  scene.background.setRGB(.75, .625, .25);
  const second = generator.fromScene(scene, 0, .1, 100, {size: 16}), secondBox = generator._backgroundBox;
  const secondDisposal = observeDisposal([secondBox.geometry, secondBox.material]);
  t.after(() => { first.dispose(); second.dispose(); });
  assert.notEqual(firstBox, secondBox); firstDisposal(0); secondDisposal(0);
  generator.dispose(); scene.background.setRGB(0, 0, 0); await renderer.waitForGPU();
  firstDisposal(1); secondDisposal(1);
  assert.deepEqual(fixedDraws(device).map(draw => Array.from(packet(draw).slice(16, 19))),
    [[.125, .25, .5], [.75, .625, .25]]);
});

test('native queued background state survives sharp to blurred to sharp transitions and later edits', async t => {
  const {renderer, device} = fixture(t); await renderer.init();
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(), new THREE.MeshBasicMaterial({color: 0xff0000})));
  scene.background = panorama(); scene.backgroundIntensity = .375; scene.backgroundRotation.y = .25;
  renderer.render(scene, camera);
  const first = packAnimationBackgroundFrame(threeBackgroundFrame(scene, camera, THREE));
  scene.backgroundBlurriness = .5; scene.backgroundIntensity = .625; scene.backgroundRotation.y = .5;
  renderer.render(scene, camera);
  scene.backgroundBlurriness = 0; scene.backgroundIntensity = .875; scene.backgroundRotation.y = .75;
  renderer.render(scene, camera);
  const third = packAnimationBackgroundFrame(threeBackgroundFrame(scene, camera, THREE));
  scene.background = new THREE.Color(0x0000ff); scene.backgroundIntensity = 9;
  scene.backgroundBlurriness = .9; scene.backgroundRotation.y = 2; camera.rotation.y = 1;
  await renderer.waitForGPU();
  const sharp = device.snapshots.flat().filter(draw => draw.pipeline.fragment?.module?.code?.includes('struct BackgroundInfo'));
  assert.equal(sharp.length, 2); assert.deepEqual(packet(sharp[0]), first); assert.deepEqual(packet(sharp[1]), third);
  const blurred = device.snapshots.filter(draws => draws.length === 2 && draws[0].pipeline.vertex.entryPoint === 'main');
  assert.equal(blurred.length, 1); assert.equal(renderer.info.f3d.presentedRenders, 3);
  assert.equal(renderer.info.f3d.offscreenRenders, 13); assert.equal(renderer._nodeCaptureBytes, 0);
});

test('native queued Color clears retain each accepted value', async t => {
  const {renderer, device} = fixture(t); await renderer.init(); renderer.autoClear = false;
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera();
  scene.background = new THREE.Color(.125, .25, .5); renderer.render(scene, camera);
  scene.background.setRGB(.75, .625, .25); renderer.render(scene, camera);
  scene.background.setRGB(0, 0, 0); await renderer.waitForGPU();
  assert.deepEqual(device.passes.map(pass => pass.desc.colorAttachments[0].clearValue),
    [{r: .125, g: .25, b: .5, a: 1}, {r: .75, g: .625, b: .25, a: 1}]);
  assert.ok(device.passes.every(pass => pass.desc.colorAttachments[0].loadOp === 'clear'));
});

test('pre-init native rendering captures background selection with deferred node updates', async t => {
  const {renderer, device} = fixture(t), scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera();
  let removedUpdates = 0; renderer.autoClear = false;
  scene.background = new THREE.Color(0xff0000);
  scene.backgroundNode = N.uniform(new THREE.Color(0x00ff00)).onRenderUpdate(() => { removedUpdates++; });
  const rendering = renderer.renderAsync(scene, camera), request = renderer._pending[0];
  const texture = panorama(); scene.background = texture; scene.backgroundNode = null; scene.backgroundIntensity = .375;
  await rendering;
  assert.equal(removedUpdates, 0); assert.equal(request.backgroundState.background, texture);
  assert.equal(request.backgroundState.backgroundNode, null); assert.equal(request.key[2], '1');
  const draws = device.snapshots.flat(); assert.equal(draws.length, 1);
  assert.ok(draws[0].pipeline.fragment.module.code.includes('struct BackgroundInfo'));
  assert.equal(packet(draws[0])[28], .375);
  assert.ok(device.passes.every(pass => pass.desc.colorAttachments[0].loadOp === 'load'));
});

test('failed queued submission still disposes captured private PMREM resources', async () => {
  const renderer = {isF3DRenderer: true, isWebGPURenderer: true, _drain: Promise.resolve(),
    waitForGPU: () => Promise.reject(new Error('recorded device loss'))};
  const generator = new PMREMGenerator(renderer);
  generator._pingPongRenderTarget = new THREE.RenderTarget(16, 16);
  generator._blurMaterial = new THREE.NodeMaterial(); generator._ggxMaterial = new THREE.NodeMaterial();
  generator._cubemapMaterial = new THREE.NodeMaterial(); generator._equirectMaterial = new THREE.NodeMaterial();
  generator._lodMeshes = [new THREE.Mesh(new THREE.PlaneGeometry())];
  generator._backgroundBox = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshBasicMaterial());
  const disposed = observeDisposal([...filteringResources(generator), generator._cubemapMaterial,
    generator._equirectMaterial, generator._backgroundBox.geometry, generator._backgroundBox.material]);
  assert.equal(generator.dispose(), undefined); disposed(0);
  await Promise.resolve(); disposed(1);
});

test('registered scene environment scenario samples the baked sky after immediate generator disposal', async t => {
  const {renderer, device} = fixture(t); await renderer.init();
  const result = nativeTSLScenarios.native_tsl_scene_environment({...THREE, PMREMGenerator}, renderer);
  renderer.render(result.scene, result.camera); await renderer.waitForGPU();
  const surfaces = device.snapshots.flat().filter(draw =>
    draw.pipeline.fragment?.module?.code?.includes('sceneEnvironmentRoughness'));
  assert.equal(surfaces.length, 2);
  for (const draw of surfaces) {
    assert.match(draw.pipeline.fragment.module.code, /fn roughnessToMip/);
    const atlases = [...draw.groups.values()].flatMap(({group}) => group.entries)
      .map(entry => entry.resource?.texture).filter(texture =>
        texture?.format === 'rgba16float' && texture.width === result.atlas.width && texture.height === result.atlas.height);
    assert.equal(atlases.length, 1, 'each surface samples the retained scene-generated CubeUV atlas');
    assert.equal(atlases[0].destroyed, undefined);
  }
  assert.ok(renderer.info.f3d.offscreenRenders >= 20);
  assert.equal(result.scene.environment, result.atlas.texture);
  assert.equal(renderer._nodeCaptureBytes, 0);
});
