/** Retained native PMREM generation through the actual facade and command
 * owner. The recording GPU proves resource/callback/FIFO behavior, not pixels. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
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
  device.textures = []; device.panoramaUploads = [];
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
    if (destination.texture.width === 64 && destination.texture.height === 32)
      device.panoramaUploads.push({bytes: destination.texture.pixels.slice(),
        drawsBefore: device.snapshots.reduce((count, draws) => count + draws.length, 0)});
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
test('native PBR reuses its atlas and prepares requested refresh before draw callbacks', async t => {
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
  f.scene.environment.needsPMREMUpdate = true;
  assert.throws(() => programs.captureDraw(descriptor), {code: 'THREE_NODE_PROGRAM_ENVIRONMENT_REFRESH'});
  assert.equal(updates, 2, 'standalone draw capture requires the ordered preparation boundary');
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(updates, 3);
  assert.equal(f.renderer.info.f3d.offscreenRenders, 2 * initialPasses);
  assert.equal(pbrDraws(f.device).length, 3);
  assert.equal(targetTextures(pbrDraws(f.device)[2])[0], atlas);
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

test('authored equirectangular PMREM inputs prepare, replace and refresh the same atlas', async t => {
  const f = fixture(t); await f.renderer.init(); let updates = 0;
  const pmrem = N.pmremTexture(f.scene.environment, N.vec3(0, 0, 1), N.float(.375));
  const material = new THREE.NodeMaterial(); material.lights = false;
  material.colorNode = N.vec4(pmrem).mul(N.uniform(1).setName('environmentSurface').onObjectUpdate(() => { updates++; return 1; }));
  f.scene.children[0].material = material; f.scene.environment = null;
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(updates, 1); assert.equal(f.renderer.info.f3d.offscreenRenders, 13);
  const first = pbrDraws(f.device)[0], firstAtlas = targetTextures(first)[0]; assert.ok(firstAtlas);
  pmrem.value = panorama(0x3400);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(updates, 2); assert.equal(f.renderer.info.f3d.offscreenRenders, 26);
  assert.notEqual(targetTextures(pbrDraws(f.device)[1])[0], firstAtlas);
  const secondAtlas = targetTextures(pbrDraws(f.device)[1])[0];
  pmrem.value.needsPMREMUpdate = true;
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(updates, 3);
  assert.equal(targetTextures(pbrDraws(f.device)[2])[0], secondAtlas);
  assert.equal(f.renderer.info.f3d.offscreenRenders, 39); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('authored PMREM size changes refuse before retiring queued filter targets', async t => {
  const f = fixture(t); await f.renderer.init(); let updates = 0;
  const pmrem = N.pmremTexture(f.scene.environment, N.vec3(0, 0, 1), N.float(.375));
  const material = new THREE.NodeMaterial(); material.lights = false;
  material.colorNode = N.vec4(pmrem).mul(N.uniform(1).setName('environmentSurface').onObjectUpdate(() => { updates++; return 1; }));
  f.scene.children[0].material = material; f.scene.environment = null;
  f.renderer.render(f.scene, f.camera);
  const pending = f.renderer._pending.length, submissions = f.device.submissions.length;
  let disposed = 0;
  pmrem._generator._pingPongRenderTarget.addEventListener('dispose', () => { disposed++; });
  const larger = panorama(0x3400);
  larger.image = {data: new Uint16Array(128 * 64 * 4).fill(0x3400), width: 128, height: 64};
  larger.needsUpdate = true; pmrem.value = larger;
  assert.throws(() => f.renderer.render(f.scene, f.camera), error =>
    error.code === 'THREE_NODE_PROGRAM_ENVIRONMENT_REFRESH' && /new PMREMNode/.test(error.message));
  assert.equal(updates, 1); assert.equal(disposed, 0);
  assert.equal(f.renderer._pending.length, pending); assert.equal(f.device.submissions.length, submissions);
  await drain(f.renderer);
  assert.equal(f.renderer.info.f3d.offscreenRenders, 13); assert.equal(pbrDraws(f.device).length, 1);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('queued in-place PMREM refresh stays between earlier and later draws with captured source uploads', async t => {
  const f = fixture(t); await f.renderer.init(); const contexts = [];
  f.material.colorNode.onObjectUpdate(frame => {
    contexts.push([frame.object, frame.material, frame.scene, frame.camera, frame.renderId]);
    return new THREE.Color(.5, .25, .125);
  });
  f.renderer.render(f.scene, f.camera);
  f.scene.environment.image.data.fill(0x3400);
  f.scene.environment.needsUpdate = true; f.scene.environment.needsPMREMUpdate = true;
  f.renderer.render(f.scene, f.camera);
  assert.equal(contexts.length, 2);
  assert.deepEqual(contexts.map(([object, material, scene, camera]) =>
    [object === f.scene.children[0], material === f.material, scene === f.scene, camera === f.camera]),
    [[true, true, true, true], [true, true, true, true]]);
  assert.notEqual(contexts[0][4], contexts[1][4]);
  await drain(f.renderer);
  assert.equal(contexts.length, 2, 'asynchronous preparation never repeats source updates');
  const all = f.device.snapshots.flat(), draws = pbrDraws(f.device);
  assert.equal(draws.length, 2); assert.equal(f.renderer.info.f3d.offscreenRenders, 26);
  const firstIndex = all.indexOf(draws[0]), secondIndex = all.indexOf(draws[1]);
  assert.ok(firstIndex >= 13); assert.ok(secondIndex - firstIndex > 13);
  assert.equal(targetTextures(draws[0])[0], targetTextures(draws[1])[0]);
  assert.deepEqual(f.device.panoramaUploads.map(({bytes}) =>
    new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2)[0]), [0x3800, 0x3400]);
  assert.ok(f.device.panoramaUploads[0].drawsBefore < firstIndex);
  assert.ok(f.device.panoramaUploads[1].drawsBefore > firstIndex);
  assert.ok(f.device.panoramaUploads[1].drawsBefore < secondIndex);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('refresh refuses before an authored before-update that precedes its source PMREM node', async t => {
  const f = fixture(t); await f.renderer.init(); let beforeCalls = 0;
  const before = N.uniform(1).setName('environmentSurface');
  before.updateBeforeType = 'object'; before.updateBefore = () => { beforeCalls++; };
  const pmrem = N.pmremTexture(f.scene.environment, N.vec3(0, 0, 1), N.float(.375));
  const material = new THREE.NodeMaterial(); material.lights = false;
  material.colorNode = N.vec4(before).mul(pmrem);
  f.scene.children[0].material = material; f.scene.environment = null;
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(beforeCalls, 1);
  const submissions = f.device.submissions.length;
  pmrem.value.needsPMREMUpdate = true;
  assert.throws(() => f.renderer.render(f.scene, f.camera), error =>
    error.code === 'THREE_NODE_PROGRAM_ENVIRONMENT_REFRESH' && /begin with/.test(error.message));
  assert.equal(beforeCalls, 1); assert.equal(f.device.submissions.length, submissions);
  assert.equal(f.renderer.info.f3d.offscreenRenders, 13); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('a callback cannot regenerate PMREM inside capture or invalidate an earlier accepted draw', async t => {
  const f = fixture(t); await f.renderer.init(); let updates = 0;
  f.material.colorNode.onObjectUpdate(() => {
    if (++updates === 2) f.scene.environment.needsPMREMUpdate = true;
    return new THREE.Color(.5, .25, .125);
  });
  f.renderer.render(f.scene, f.camera);
  const pending = f.renderer._pending.length;
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_ENVIRONMENT_REFRESH'});
  assert.equal(updates, 2); assert.equal(f.renderer._pending.length, pending);
  await drain(f.renderer);
  assert.equal(f.renderer.info.f3d.offscreenRenders, 13); assert.equal(pbrDraws(f.device).length, 1);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('a later draw cannot move its independent source refresh ahead of earlier draw updates', async t => {
  const f = fixture(t); await f.renderer.init(); f.renderer.sortObjects = false; let updates = 0;
  const secondSource = panorama(0x3400);
  const makeMaterial = texture => {
    const material = new THREE.NodeMaterial(); material.lights = false;
    material.colorNode = N.vec4(N.pmremTexture(texture, N.vec3(0, 0, 1), N.float(.375))).mul(
      N.uniform(1).setName('environmentSurface').onObjectUpdate(() => { updates++; return 1; }));
    return material;
  };
  f.scene.children[0].material = makeMaterial(f.scene.environment);
  f.scene.add(new THREE.Mesh(new THREE.PlaneGeometry(), makeMaterial(secondSource))); f.scene.environment = null;
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(updates, 2);
  const submissions = f.device.submissions.length;
  secondSource.needsPMREMUpdate = true;
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_ENVIRONMENT_REFRESH'});
  assert.equal(updates, 2); assert.equal(f.device.submissions.length, submissions);
  assert.equal(f.renderer.info.f3d.offscreenRenders, 26); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('a stale PMREM graph recompile refuses before source setup can enqueue filter draws', async t => {
  const f = fixture(t); await f.renderer.init(); let updates = 0;
  f.material.colorNode.onObjectUpdate(() => { updates++; return new THREE.Color(.5, .25, .125); });
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const submissions = f.device.submissions.length, programs = f.renderer._nodeProgramSupport.diagnostics.programs;
  f.scene.environment.needsPMREMUpdate = true; f.material.needsUpdate = true;
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_ENVIRONMENT_REFRESH'});
  assert.equal(updates, 1); assert.equal(f.device.submissions.length, submissions);
  assert.equal(f.renderer._nodeProgramSupport.diagnostics.programs, programs);
  assert.equal(f.renderer.info.f3d.offscreenRenders, 13); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('accepted refresh filters survive a later outer callback failure and remain reusable', async t => {
  const f = fixture(t); await f.renderer.init(); let refuse = false, updates = 0;
  f.material.colorNode.onObjectUpdate(() => {
    updates++;
    if (refuse) throw new Error('authored draw refused');
    return new THREE.Color(.5, .25, .125);
  });
  f.renderer.render(f.scene, f.camera);
  f.scene.environment.image.data.fill(0x3400);
  f.scene.environment.needsUpdate = true; f.scene.environment.needsPMREMUpdate = true; refuse = true;
  assert.throws(() => f.renderer.render(f.scene, f.camera), /authored draw refused/);
  await drain(f.renderer);
  assert.equal(f.renderer.info.f3d.offscreenRenders, 26); assert.equal(pbrDraws(f.device).length, 1);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
  refuse = false; f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(updates, 3); assert.equal(f.renderer.info.f3d.offscreenRenders, 26);
  assert.equal(pbrDraws(f.device).length, 2); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('mixed fixed atlas readers keep their earlier queued draw when native refresh would precede them', async t => {
  const f = fixture(t); await f.renderer.init(); let updates = 0;
  const pmrem = N.pmremTexture(f.scene.environment, N.vec3(0, 0, 1), N.float(.375));
  const material = new THREE.NodeMaterial(); material.lights = false;
  material.colorNode = N.vec4(pmrem).mul(
    N.uniform(1).setName('environmentSurface').onObjectUpdate(() => { updates++; return 1; }));
  const object = f.scene.children[0]; object.material = material; f.scene.environment = null;
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const atlas = targetTextures(pbrDraws(f.device)[0])[0];
  const fixed = new THREE.Mesh(new THREE.PlaneGeometry(), new THREE.MeshBasicMaterial({map: pmrem._pmrem}));
  fixed.position.z = 1; fixed.renderOrder = -1; f.scene.add(fixed);
  // Keep an accepted fixed atlas reader queued, then select both routes for
  // the refused refresh. Its earlier packet must retain the old atlas.
  object.visible = false; f.renderer.render(f.scene, f.camera);
  assert.equal(updates, 1);
  const pending = f.renderer._pending.length, submissions = f.device.submissions.length;
  object.visible = true; pmrem.value.needsPMREMUpdate = true;
  assert.throws(() => f.renderer.render(f.scene, f.camera), error =>
    error.code === 'THREE_NODE_PROGRAM_ENVIRONMENT_REFRESH' && /fixed scene draws/.test(error.message));
  assert.equal(updates, 1); assert.equal(f.renderer._pending.length, pending);
  assert.equal(f.device.submissions.length, submissions);
  await drain(f.renderer);
  assert.equal(f.renderer.info.f3d.offscreenRenders, 13); assert.equal(pbrDraws(f.device).length, 1);
  const accepted = f.device.snapshots.at(-1);
  assert.equal(accepted.length, 1);
  assert.ok(!accepted[0].pipeline.fragment.module.code.includes('environmentSurface'));
  const orientation = f.device.passes.findLast(pass =>
    pass.draws.some(draw => draw.pipeline.label === 'Source target UV orientation'));
  assert.ok(orientation);
  assert.equal(targetTextures(orientation.draws[0])[0], atlas);
  assert.equal(orientation.desc.colorAttachments[0].view.texture, targetTextures(accepted[0])[0]);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('a private native background keeps refresh before fixed and native scene draws', async t => {
  const f = fixture(t); await f.renderer.init(); let updates = 0;
  const source = f.scene.environment;
  f.scene.background = source; f.scene.backgroundBlurriness = .5; f.scene.environment = null;
  const material = new THREE.NodeMaterial(); material.lights = false;
  material.colorNode = N.vec4(N.pmremTexture(source, N.vec3(0, 0, 1), N.float(.375))).mul(
    N.uniform(1).setName('environmentSurface').onObjectUpdate(() => { updates++; return 1; }));
  f.scene.children[0].material = material;
  const fixed = new THREE.Mesh(new THREE.PlaneGeometry(), new THREE.MeshBasicMaterial({color: 0xabcdef}));
  fixed.position.z = 1; fixed.renderOrder = -1; f.scene.add(fixed);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(updates, 2); assert.equal(f.renderer.info.f3d.offscreenRenders, 13);
  source.image.data.fill(0x3400); source.needsUpdate = true; source.needsPMREMUpdate = true;
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(updates, 3); assert.equal(f.renderer.info.f3d.offscreenRenders, 26);
  assert.equal(pbrDraws(f.device).length, 3); assert.equal(f.renderer._nodeCaptureBytes, 0);
});
