/** Retained native PMREM generation through the actual facade and command
 * owner. The recording GPU proves resource/callback/FIFO behavior, not pixels. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import {nativeTSLScenarios} from '../../tests/e2e/webgpu_renderer/scenarios_native_tsl.js';
const Renderer = createWebGPURendererClass(THREE, {maxDraws: 64, maxBindings: 64}), N = THREE.TSL;
function panorama(value = 0x3800) {
  const texture = new THREE.DataTexture(new Uint16Array(64 * 32 * 4).fill(value), 64, 32,
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
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  const material = new THREE.MeshStandardNodeMaterial({metalness: .25, roughness: .5});
  material.colorNode = N.uniform(new THREE.Color(.5, .25, .125)).setName('environmentSurface');
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(), material)); scene.environment = panorama();
  return {renderer, device, scene, camera, material};
}
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
const pbrDraws = device => device.snapshots.flat().filter(draw => draw.pipeline.fragment?.module?.code?.includes('environmentSurface'));
const targetTextures = draw => [...draw.groups.values()].flatMap(({group}) => group.entries)
  .map(entry => entry.resource?.texture).filter(texture => texture?.format === 'rgba16float');
function uniformValues(draw) {
  return [...draw.groups.values()].flatMap(({group, offsets}) => group.entries.flatMap(({resource}) => {
    if (!resource.buffer) return [];
    const bytes = draw.contents.get(resource.buffer);
    const offset = (resource.offset ?? 0) + (offsets[0] ?? 0);
    return [...new Float32Array(bytes.buffer, bytes.byteOffset + offset, resource.size / 4)];
  }));
}

test('blurred source background shares the PBR atlas and captures queued source controls', async t => {
  const f = fixture(t); await f.renderer.init();
  f.scene.background = f.scene.environment;
  f.scene.backgroundBlurriness = .25; f.scene.backgroundIntensity = .375; f.scene.backgroundRotation.y = .25;
  const children = f.scene.children.slice();
  f.renderer.render(f.scene, f.camera);
  f.scene.backgroundBlurriness = .625; f.scene.backgroundIntensity = .875; f.scene.backgroundRotation.y = .75;
  f.renderer.render(f.scene, f.camera);
  f.scene.backgroundIntensity = 3; f.scene.backgroundRotation.y = 1.5;
  await drain(f.renderer);
  assert.deepEqual(f.scene.children, children);
  assert.equal(f.renderer.info.f3d.offscreenRenders, 13);
  const frames = f.device.snapshots.filter(draws => draws.some(draw => draw.pipeline.fragment?.module?.code?.includes('environmentSurface')));
  assert.equal(frames.length, 2);
  for (const draws of frames) {
    assert.equal(draws.length, 2);
    assert.equal(draws[0].pipeline.primitive.frontFace, 'cw');
    assert.equal(draws[0].pipeline.primitive.cullMode, 'back');
    assert.equal(draws[0].pipeline.depthStencil.depthWriteEnabled, false);
    assert.ok(draws[1].pipeline.fragment.module.code.includes('environmentSurface'));
  }
  assert.ok(uniformValues(frames[0][0]).includes(.25));
  assert.ok(uniformValues(frames[0][0]).includes(.375));
  assert.ok(uniformValues(frames[1][0]).includes(.625));
  assert.ok(uniformValues(frames[1][0]).includes(.875));
  assert.ok(uniformValues(frames[0][0]).some(value => Math.abs(value - Math.cos(.25)) < 1e-6));
  assert.ok(uniformValues(frames[1][0]).some(value => Math.abs(value - Math.cos(.75)) < 1e-6));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});
test('blurred background draws before fixed materials with sorting disabled', async t => {
  const f = fixture(t); await f.renderer.init();
  f.scene.background = f.scene.environment; f.scene.environment = null;
  f.scene.backgroundBlurriness = .5;
  f.scene.children[0].material = new THREE.MeshBasicMaterial({color: 0xff0000});
  f.renderer.sortObjects = false;
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const draws = f.device.snapshots.at(-1);
  assert.equal(draws.length, 2);
  assert.equal(draws[0].pipeline.primitive.frontFace, 'cw');
  assert.equal(draws[0].pipeline.depthStencil.depthWriteEnabled, false);
  assert.equal(draws[1].pipeline.depthStencil.depthWriteEnabled, true);
  assert.equal(f.scene.children.length, 1); assert.equal(f.renderer._nodeCaptureBytes, 0);
});
test('queued background swaps retain differently sized atlases and reuse earlier texture identities', async t => {
  const f = fixture(t); await f.renderer.init();
  const first = f.scene.environment, second = panorama(0x3400);
  second.image = {data: new Uint16Array(128 * 64 * 4).fill(0x3400), width: 128, height: 64};
  second.needsUpdate = true;
  f.scene.environment = null; f.scene.backgroundBlurriness = .5;
  f.scene.children[0].material = new THREE.MeshBasicMaterial({color: 0xff0000});
  for (const texture of [first, second, first]) {
    f.scene.background = texture; f.renderer.render(f.scene, f.camera);
  }
  await drain(f.renderer);
  const frames = f.device.snapshots.filter(draws => draws.length === 2 && draws[0].pipeline.primitive?.frontFace === 'cw');
  assert.equal(frames.length, 3);
  const atlases = frames.map(draws => targetTextures(draws[0])[0]);
  assert.ok(atlases.every(Boolean)); assert.notEqual(atlases[0], atlases[1]); assert.equal(atlases[0], atlases[2]);
  assert.notEqual(atlases[0].height, atlases[1].height);
  assert.equal(f.scene.children.length, 1); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('authored H7 sky nodes take precedence and capture original render context and controls', async t => {
  const f = fixture(t); await f.renderer.init(); f.scene.environment = null;
  f.scene.children[0].material = new THREE.MeshBasicMaterial({color: 0xff0000});
  f.scene.background = new THREE.CubeTexture();
  const bottom = N.uniform(new THREE.Color(.125, .25, .5)).setName('authoredSky');
  let updates = 0, disposed = 0;
  bottom.onRenderUpdate(frame => {
    assert.equal(frame.scene, f.scene); assert.equal(frame.camera, f.camera);
    assert.equal(f.scene.children.includes(frame.object), false); updates++;
    return bottom.value;
  });
  const sky = N.screenUV.y.mix(bottom, N.color(0xe7eaeb));
  sky.addEventListener('dispose', () => { disposed++; });
  f.scene.backgroundNode = sky; f.scene.backgroundIntensity = .375;
  const children = f.scene.children.slice();
  f.renderer.render(f.scene, f.camera);
  bottom.value.setRGB(.75, .625, .25); f.scene.backgroundIntensity = .875;
  f.renderer.render(f.scene, f.camera);
  bottom.value.setRGB(0, 0, 0); f.scene.backgroundIntensity = 9;
  await drain(f.renderer);
  const frames = f.device.snapshots.filter(draws => draws[0]?.pipeline.fragment?.module?.code?.includes('authoredSky'));
  assert.equal(frames.length, 2); assert.ok(frames.every(draws => draws.length === 2));
  assert.equal(updates, 2);
  assert.ok(uniformValues(frames[0][0]).includes(.125)); assert.ok(uniformValues(frames[0][0]).includes(.375));
  assert.ok(uniformValues(frames[1][0]).includes(.75)); assert.ok(uniformValues(frames[1][0]).includes(.875));
  assert.equal(f.renderer.info.f3d.offscreenRenders, 0);
  assert.deepEqual(f.scene.children, children); assert.equal(f.renderer._nodeCaptureBytes, 0);
  f.renderer.dispose(); assert.equal(disposed, 0, 'renderer does not own the authored background node');
});
test('authored sky replacement preserves queued graphs and overrides a color without forcing a clear', async t => {
  const f = fixture(t); await f.renderer.init(); f.scene.environment = null;
  f.scene.children[0].material = new THREE.MeshBasicMaterial({color: 0xff0000});
  f.scene.background = new THREE.Color(0xffffff); f.renderer.autoClear = false;
  f.scene.backgroundNode = N.uniform(new THREE.Color(.125, .25, .5)).setName('firstSky');
  f.renderer.render(f.scene, f.camera);
  f.scene.backgroundNode = N.uniform(new THREE.Color(.75, .625, .25)).setName('secondSky');
  f.renderer.render(f.scene, f.camera);
  await drain(f.renderer);
  assert.equal(f.device.snapshots.length, 2);
  assert.ok(f.device.snapshots[0][0].pipeline.fragment.module.code.includes('firstSky'));
  assert.ok(f.device.snapshots[1][0].pipeline.fragment.module.code.includes('secondSky'));
  assert.ok(f.device.passes.every(pass => pass.desc.colorAttachments[0].loadOp === 'load'));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});
test('unsupported authored background resources fail before callbacks or submission', async t => {
  const f = fixture(t); await f.renderer.init(); let updates = 0;
  f.scene.environment = null;
  f.scene.backgroundNode = N.texture(new THREE.CubeTexture());
  f.material.colorNode.onObjectUpdate(() => { updates++; return new THREE.Color(.5, .25, .125); });
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_TEXTURE'});
  assert.equal(updates, 0); assert.equal(f.device.submissions.length, 0);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});


test('registered blurred-background scenario queues three independent target captures', async t => {
  const f = fixture(t); await f.renderer.init();
  const scenario = nativeTSLScenarios.native_tsl_background_blur(THREE, f.renderer);
  f.renderer.render(scenario.scene, scenario.camera); await drain(f.renderer);
  const frames = f.device.snapshots.filter(draws => draws.length === 2 && draws[0].pipeline.primitive?.frontFace === 'cw');
  assert.equal(frames.length, 3);
  for (const [index, blur] of [.08, .72, .32].entries())
    assert.ok(uniformValues(frames[index][0]).some(value => Math.abs(value - blur) < 1e-6));
  assert.ok(uniformValues(frames[2][0]).some(value => Math.abs(value - .42) < 1e-6));
  assert.equal(f.renderer.info.f3d.offscreenRenders, 16);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});
