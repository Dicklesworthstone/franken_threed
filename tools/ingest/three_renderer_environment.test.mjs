/** Retained native PMREM generation through the actual facade and command
 * owner. The recording GPU proves resource/callback/FIFO behavior, not pixels. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
const Renderer = createWebGPURendererClass(THREE), N = THREE.TSL;
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
test('native PBR queues the source PMREM atlas before its draw and refuses in-place refresh before callbacks', async t => {
  const f = fixture(t); await f.renderer.init(); let updates = 0;
  f.material.colorNode.onObjectUpdate(() => { updates++; return new THREE.Color(.5, .25, .125); });
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(pbrDraws(f.device).length, 1);
  const initialPasses = f.renderer.info.f3d.offscreenRenders;
  assert.equal(initialPasses, 13, 'the pinned 16px cube faces produce one conversion and twelve filter draws');
  const first = pbrDraws(f.device)[0], atlas = targetTextures(first)[0]; assert.ok(atlas);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(f.renderer.info.f3d.offscreenRenders, initialPasses);
  assert.equal(targetTextures(pbrDraws(f.device)[1])[0], atlas);
  assert.equal(updates, 2);
  const programs = f.renderer._nodeProgramSupport;
  const descriptor = programs.compile(f.material, f.scene.children[0], {scene: f.scene, camera: f.camera});
  const submissions = f.device.submissions.length;
  f.scene.environment.needsPMREMUpdate = true;
  assert.throws(() => programs.captureDraw(descriptor), {code: 'THREE_NODE_PROGRAM_ENVIRONMENT_REFRESH'});
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_ENVIRONMENT_REFRESH'});
  assert.equal(updates, 2, 'unsupported refresh fails before authored node callbacks');
  assert.equal(f.renderer.info.f3d.offscreenRenders, initialPasses);
  assert.equal(f.device.submissions.length, submissions);
  assert.equal(pbrDraws(f.device).length, 2);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});
test('queued source environment replacement keeps atlas identity and source uniform updates per call', async t => {
  const f = fixture(t); await f.renderer.init(); let updates = 0;
  f.material.colorNode.onObjectUpdate(() => { updates++; return new THREE.Color(.5, .25, .125); });
  f.scene.environmentIntensity = .375;
  f.renderer.render(f.scene, f.camera);
  f.scene.environment = panorama(0x3400); f.scene.environmentIntensity = .875;
  f.renderer.render(f.scene, f.camera);
  f.scene.environmentIntensity = 2;
  assert.equal(updates, 2); await drain(f.renderer); assert.equal(updates, 2);
  const draws = pbrDraws(f.device); assert.equal(draws.length, 2);
  assert.notEqual(targetTextures(draws[0])[0], targetTextures(draws[1])[0]);
  assert.ok(uniformValues(draws[0]).includes(.375)); assert.ok(uniformValues(draws[1]).includes(.875));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});
