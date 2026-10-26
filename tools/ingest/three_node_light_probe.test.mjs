/** Pinned spherical-harmonic lighting and F3D queued uniform submission.
 * The recording device verifies bytes and ownership, not GPU pixel parity. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {captureThreeNodeScene} from './three_node_scene_capture.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = THREE.TSL, SH_BYTES = 9 * 4 * 4;
const shShader = /array<\s*vec4<f32>,\s*9\s*>/;
function setCoefficients(probe, scale = 1) {
  probe.sh.coefficients.forEach((coefficient, index) =>
    coefficient.set(scale * (index + 1) / 64, scale * (index + 2) / 64, scale * (index + 3) / 64));
}
const expectedSH = probe => new Float32Array(probe.sh.coefficients.flatMap(coefficient =>
  [coefficient.x * probe.intensity, coefficient.y * probe.intensity, coefficient.z * probe.intensity, 0]));
function uniformSH(descriptor, packet) {
  const blocks = descriptor.groups.flatMap((group, index) => group.uniforms.filter(uniform => uniform.byteLength === SH_BYTES &&
    group.entries.find(entry => entry.binding === uniform.binding).visibility === 2)
    .map(uniform => new Float32Array(packet.uniforms[index].buffer,
      packet.uniforms[index].byteOffset + uniform.offset, SH_BYTES / 4)));
  assert.equal(blocks.length, 1); return blocks[0];
}
function sourceRig(t, Material = THREE.MeshStandardMaterial) {
  const frame = new THREE.NodeFrame(); frame.frameId = 1;
  const renderer = {library: new THREE.StandardNodeLibrary(), contextNode: N.context(),
    debug: {diagnostics: {keywords: false}}, depth: true, lighting: {enabled: true}, sortObjects: true,
    backend: {hasFeature: () => false, utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})}},
    hasFeature: () => false, getRenderTarget: () => null, getMRT: () => null, currentSamples: 1,
    _nodes: {nodeFrame: frame}, info: {calls: 1}};
  const programs = createThreeNodePrograms({three: THREE, renderer}); t.after(() => programs.dispose());
  const material = new Material(), object = new THREE.Mesh(new THREE.PlaneGeometry(), material);
  const probe = new THREE.LightProbe(); setCoefficients(probe); probe.intensity = .5;
  const scene = new THREE.Scene(); scene.add(object, probe);
  const camera = new THREE.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  camera.coordinateSystem = THREE.WebGPUCoordinateSystem; camera.updateProjectionMatrix();
  scene.updateMatrixWorld(); camera.updateMatrixWorld();
  const context = {object, material, scene, camera, lights: [probe], frameId: 1, renderId: 1};
  const compile = () => programs.compile(material, object, context);
  const capture = options => captureThreeNodeScene(THREE, scene, camera, {renderer,
    getPrograms: () => programs, raster: material => ({side: material.side}),
    getTextures: () => ({captureBinding: binding => binding, release() {},
      captureDispatch: run => run(), validateDispatch() {}}), ...options});
  return {programs, renderer, frame, material, object, scene, probe, camera, context, compile, capture};
}

for (const Material of [THREE.MeshToonMaterial, THREE.MeshLambertMaterial, THREE.MeshPhongMaterial, THREE.MeshStandardMaterial, THREE.MeshPhysicalMaterial,
  THREE.MeshToonNodeMaterial, THREE.MeshLambertNodeMaterial, THREE.MeshPhongNodeMaterial, THREE.MeshStandardNodeMaterial, THREE.MeshPhysicalNodeMaterial]) {
  test(`${Material.name} probe lighting matches the pinned shader and all nine padded SH coefficients`, t => {
    const f = sourceRig(t, Material), reference = new THREE.WGSLNodeBuilder(f.object, f.renderer);
    reference.material = f.material; reference.context.material = f.material;
    reference.scene = f.scene; reference.camera = f.camera; reference.lightsNode = N.lights([f.probe]); reference.build();
    const descriptor = f.compile(), packet = f.programs.captureDraw(descriptor, f.context);
    assert.equal(descriptor.vertex, reference.vertexShader); assert.equal(descriptor.fragment, reference.fragmentShader);
    assert.match(descriptor.fragment, shShader);
    assert.deepEqual(uniformSH(descriptor, packet), expectedSH(f.probe));
    assert.ok(f.capture().objects.has(f.object), 'ordinary probe-lit materials automatically select the native owner');
    assert.equal(Object.hasOwn(f.material, 'colorNode') && f.material.colorNode !== null, false,
      'ordinary materials need no authored graph to consume probe irradiance');
  });
}

test('shared probes update once per source frame across objects, renders and program rebuilds', t => {
  const f = sourceRig(t), descriptor = f.compile(), first = f.programs.captureDraw(descriptor, f.context);
  const saved = uniformSH(descriptor, first).slice();
  f.probe.intensity = .75; setCoefficients(f.probe, 2);
  const other = new THREE.Mesh(f.object.geometry, f.material); f.scene.add(other); f.scene.updateMatrixWorld();
  const otherDescriptor = f.programs.compile(f.material, other, {...f.context, object: other});
  const sameFrame = f.programs.captureDraw(otherDescriptor, {...f.context, object: other, renderId: 2});
  assert.deepEqual(uniformSH(otherDescriptor, sameFrame), saved);
  f.material.needsUpdate = true;
  const rebuilt = f.compile(); assert.notEqual(rebuilt, descriptor);
  assert.deepEqual(uniformSH(rebuilt, f.programs.captureDraw(rebuilt, {...f.context, renderId: 3})), saved);
  const nextFrame = f.programs.captureDraw(rebuilt, {...f.context, frameId: 2, renderId: 4});
  assert.deepEqual(uniformSH(rebuilt, nextFrame), expectedSH(f.probe));
  assert.deepEqual(uniformSH(descriptor, first), saved, 'accepted coefficient bytes stay immutable');
  assert.equal(f.compile(), rebuilt, 'coefficient and intensity updates do not change shader structure');
});

test('zero-intensity probes keep the source graph with zero coefficients and unlit Basic ignores them', t => {
  const f = sourceRig(t); f.probe.intensity = 0;
  const descriptor = f.compile(), packet = f.programs.captureDraw(descriptor, f.context);
  assert.deepEqual(uniformSH(descriptor, packet), new Float32Array(36));
  const basic = new THREE.MeshBasicMaterial({color: 0xabcdef}), object = new THREE.Mesh(f.object.geometry, basic);
  const unlit = f.programs.compile(basic, object, {...f.context, object, material: basic});
  assert.doesNotMatch(unlit.fragment, shShader);
});

test('inactive probes keep ArrayCamera scenes on fixed shaders without constructing native programs', t => {
  for (const inactive of ['hidden', 'layer', 'disabled']) {
    const f = sourceRig(t), camera = new THREE.ArrayCamera([f.camera]);
    if (inactive === 'hidden') {
      const parent = new THREE.Group(); parent.visible = false; parent.add(f.probe); f.scene.add(parent);
    } else if (inactive === 'layer') f.probe.layers.set(1);
    else f.renderer.lighting.enabled = false;
    const captured = captureThreeNodeScene(THREE, f.scene, camera, {renderer: f.renderer,
      getPrograms() { assert.fail('inactive probes must not create a native compiler'); }});
    assert.equal(captured.bindings.length, 0); assert.equal(captured.draws.length, 0);
    assert.equal(captured.objects.has(f.object), false);
    assert.equal(captured.nativeLights.has(f.probe), true, 'the fixed owner receives the captured inactive-light decision');
  }
});

test('unlit scenes do not compile probe graphs and unsupported lit subclasses still refuse', t => {
  const f = sourceRig(t, THREE.MeshBasicMaterial);
  const captured = f.capture({getPrograms() { assert.fail('Basic has no probe lighting'); }});
  assert.equal(captured.bindings.length, 0); assert.equal(captured.objects.has(f.object), false);
  class CustomStandard extends THREE.MeshStandardMaterial {}
  f.object.material = new CustomStandard();
  assert.throws(() => f.capture(), {code: 'THREE_NODE_PROGRAM_MATERIAL'});
});

function recording(t) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => ({dimension: '2d', sampleCount: 1, ...descriptor,
    width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
    depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = descriptor => ({...descriptor}); device.queue.writeTexture = () => {};
  let configuration;
  const canvas = {width: 16, height: 16, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [16, 16], format: configuration.format, usage: 16}); }
  })};
  const Renderer = createWebGPURendererClass(THREE, {maxDraws: 32, maxBindings: 32});
  const renderer = new Renderer({canvas, device}); t.after(() => renderer.dispose());
  const material = new THREE.MeshStandardMaterial(), scene = new THREE.Scene();
  const object = new THREE.Mesh(new THREE.PlaneGeometry(), material); scene.add(object);
  const probe = new THREE.LightProbe(); setCoefficients(probe); scene.add(probe);
  const camera = new THREE.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  return {device, renderer, material, object, scene, probe, camera};
}
const isSceneDraw = draw => !draw.pipeline.label?.startsWith('Animation HDR output ');
const allDraws = device => device.snapshots.flat().filter(isSceneDraw);
function submittedSH(draw) {
  const found = [];
  for (const {group, offsets} of draw.groups.values()) {
    const buffers = group.entries.filter(entry => entry.resource?.buffer);
    for (const [index, {resource}] of buffers.entries()) if (resource.size === SH_BYTES) {
      const bytes = draw.contents.get(resource.buffer), start = (resource.offset ?? 0) + (offsets[index] ?? 0);
      found.push(new Float32Array(bytes.buffer, bytes.byteOffset + start, SH_BYTES / 4));
    }
  }
  assert.equal(found.length, 1); return found[0];
}

test('ordinary probe-lit draws retain queued coefficient versions and return to fixed shaders after removal', async t => {
  const f = recording(t); await f.renderer.init();
  f.renderer._advanceAnimationFrame();
  const first = expectedSH(f.probe); f.renderer.render(f.scene, f.camera);
  setCoefficients(f.probe, 2); f.probe.intensity = .75; f.renderer._advanceAnimationFrame();
  const second = expectedSH(f.probe); f.renderer.render(f.scene, f.camera);
  assert.notDeepEqual(first, second);
  f.scene.remove(f.probe); f.renderer.render(f.scene, f.camera);
  setCoefficients(f.probe, 99);
  await f.renderer.waitForGPU();
  const draws = allDraws(f.device); assert.equal(draws.length, 3);
  assert.match(draws[0].pipeline.fragment.module.code, shShader); assert.match(draws[1].pipeline.fragment.module.code, shShader);
  assert.deepEqual(submittedSH(draws[0]), first); assert.deepEqual(submittedSH(draws[1]), second);
  assert.equal(draws[2].pipeline.vertex.entryPoint, 'vertex_main');
  assert.doesNotMatch(draws[2].pipeline.fragment.module.code, shShader);
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.renderer._materialCaptureValues, 0);
  assert.equal(f.material.colorNode, undefined);
});

test('native probe meshes share scenes with fixed sprites, lines, points and unlit probe children', async t => {
  const f = recording(t); await f.renderer.init();
  const geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-.5, 0, 0), new THREE.Vector3(.5, 0, 0)]);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial()), line = new THREE.Line(geometry, new THREE.LineBasicMaterial());
  const points = new THREE.Points(geometry, new THREE.PointsMaterial({size: 1}));
  const basic = new THREE.Mesh(f.object.geometry, new THREE.MeshBasicMaterial());
  f.scene.add(sprite, points); f.probe.add(line, basic);
  f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
  const draws = allDraws(f.device); assert.equal(draws.length, 5);
  assert.equal(draws.filter(draw => shShader.test(draw.pipeline.fragment.module.code)).length, 1);
  assert.equal(draws.filter(draw => draw.pipeline.vertex.entryPoint === 'vertex_main').length, 4);
  assert.deepEqual(submittedSH(draws.find(draw => shShader.test(draw.pipeline.fragment.module.code))), expectedSH(f.probe));
});

test('queued probe selection retains camera layers and lighting enablement while preserving probe children', async t => {
  const f = recording(t); await f.renderer.init();
  const child = new THREE.Mesh(f.object.geometry, new THREE.MeshBasicMaterial()); f.probe.add(child);
  f.probe.layers.set(1); f.renderer.render(f.scene, f.camera);
  f.probe.layers.set(0); f.renderer.render(f.scene, f.camera);
  f.renderer.lighting.enabled = false; f.renderer.render(f.scene, f.camera);
  f.renderer.lighting.enabled = true; f.probe.layers.set(2);
  await f.renderer.waitForGPU();
  const frames = f.device.snapshots.map(draws => draws.filter(isSceneDraw)).filter(draws => draws.length);
  assert.deepEqual(frames.map(draws => draws.length), [2, 2, 2], 'probe layer filtering does not hide its ordinary children');
  assert.deepEqual(frames.map(draws => draws.filter(draw => shShader.test(draw.pipeline.fragment.module.code)).length), [0, 1, 0]);
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.renderer._materialCaptureValues, 0);
});

test('layer-excluded and disabled probes render fixed ArrayCamera views through the facade', async t => {
  const f = recording(t); await f.renderer.init();
  const camera = new THREE.ArrayCamera([f.camera]);
  f.probe.layers.set(1); f.renderer.render(f.scene, camera);
  f.probe.layers.set(0); f.renderer.lighting.enabled = false; f.renderer.render(f.scene, camera);
  f.renderer.lighting.enabled = true;
  await f.renderer.waitForGPU();
  const draws = allDraws(f.device); assert.equal(draws.length, 2);
  assert.ok(draws.every(draw => draw.pipeline.vertex.entryPoint === 'vertex_main'));
  assert.ok(draws.every(draw => !shShader.test(draw.pipeline.fragment.module.code)));
});

test('probe lighting shares native shadow ownership and cannot shift fixed shadow light indices', async t => {
  for (const lit of [false, true]) {
    const f = recording(t); await f.renderer.init();
    if (!lit) f.object.material = new THREE.MeshBasicMaterial();
    f.renderer.shadowMap.enabled = true; f.renderer.shadowMap.type = THREE.BasicShadowMap;
    f.object.receiveShadow = true;
    const light = new THREE.DirectionalLight(); light.position.set(1, 2, 3); light.castShadow = true;
    light.shadow.mapSize.set(16, 16); f.scene.add(light);
    const caster = new THREE.Mesh(new THREE.BoxGeometry(.5, .5, .5), new THREE.MeshBasicMaterial());
    caster.castShadow = true; caster.frustumCulled = false; f.scene.add(caster);
    f.renderer.render(f.scene, f.camera); await f.renderer.waitForGPU();
    const diagnostics = f.renderer._dispatcher.entry(f.scene).bridge.diagnostics;
    if (lit) {
      assert.equal(diagnostics.shadowBytes, 0, 'probe-promoted materials share the native shadow owner with fixed casters');
      assert.equal(diagnostics.shadowStats, null);
      assert.ok(allDraws(f.device).some(draw => shShader.test(draw.pipeline.fragment?.module.code ?? '')));
      assert.ok(allDraws(f.device).some(draw => /texture_depth_2d/.test(draw.pipeline.fragment?.module.code ?? '')));
    } else {
      assert.ok(diagnostics.shadowBytes > 0);
      assert.equal(diagnostics.shadowStats.lightIndex, 0, 'the skipped probe cannot offset the directional light');
      assert.equal(diagnostics.shadowStats.casters, 1);
    }
  }
});
