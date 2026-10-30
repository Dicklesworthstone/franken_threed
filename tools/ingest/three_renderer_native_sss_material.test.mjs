/** Pinned subsurface material construction and queued owned WebGPU draws.
 * The recording device checks commands and bytes, not raster pixel parity. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL, Renderer = createWebGPURendererClass(T);
const drain = async renderer => { while (renderer._drain) await renderer._drain; };
const native = f => f.device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main' && draw.streams.size);
const captures = f => f.captures.flatMap(value => value.draws ?? []).filter(draw => draw.object === f.object);
const values = packets => packets.flatMap(bytes => [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);
const bindings = draw => [...draw.groups.values()].flatMap(({group}) => group.entries);
const sssSlots = ['thicknessColorNode', 'thicknessDistortionNode', 'thicknessAmbientNode',
  'thicknessAttenuationNode', 'thicknessPowerNode', 'thicknessScaleNode'];

function fixture(t) {
  const device = geometryDevice(), textures = [], writes = [], copies = [], captured = [];
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256});
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, mipLevelCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }};
    textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor});
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  device.queue.writeTexture = (destination, data) => writes.push({texture: destination.texture,
    bytes: new Uint8Array(data.buffer ?? data, data.byteOffset ?? 0, data.byteLength).slice()});
  const createEncoder = device.createCommandEncoder;
  device.createCommandEncoder = () => {
    const encoder = createEncoder(), commands = [];
    return {...encoder, beginRenderPass(descriptor) {
      const pass = encoder.beginRenderPass(descriptor); commands.push(device.passes.at(-1)); return pass;
    }, copyTextureToTexture(source, destination, size) {
      const copy = {source, destination, size, draws: []}; copies.push(copy); commands.push(copy);
    }, finish() { return commands; }};
  };
  let configured;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 32], format: configured.format, usage: configured.usage}); },
  })};
  const renderer = new Renderer({canvas, device}); renderer.outputColorSpace = T.LinearSRGBColorSpace;
  t.after(() => renderer.dispose());
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 4;
  const material = new T.MeshSSSNodeMaterial(), object = new T.Mesh(new T.SphereGeometry(1, 8, 6), material);
  object.frustumCulled = false; scene.add(object); t.after(() => { object.geometry.dispose(); material.dispose(); });
  const capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function (request) { const value = capture.call(this, request); if (value) captured.push(value); return value; };
  return {device, renderer, scene, camera, material, object, textures, writes, copies, captures: captured};
}
function source(f, descriptor) {
  const builder = new T.WGSLNodeBuilder(f.object, f.renderer);
  builder.scene = f.scene; builder.camera = f.camera;
  builder.lightsNode = N.lights(f.scene.children.filter(object => object.isLight));
  builder.environmentNode = f.scene.environmentNode ?? (f.scene.environment ? N.texture(f.scene.environment) : null);
  builder.build();
  assert.equal(descriptor.vertex, builder.vertexShader);
  assert.equal(descriptor.fragment, builder.fragmentShader);
  return builder;
}
function controls(material) {
  const nodes = [N.uniform(new T.Color(.1875, .3125, .4375)), ...[.125, .375, .625, 2.75, 5.25].map(value => N.uniform(value))];
  sssSlots.forEach((slot, index) => { material[slot] = nodes[index].setName(slot); });
  return nodes;
}
function map(t) {
  const texture = new T.DataTexture(new Uint8Array([64, 128, 192, 255, 128, 192, 64, 255]), 2, 1);
  texture.needsUpdate = true; t.after(() => texture.dispose()); return texture;
}
function programOwner(t, f) {
  const owner = createThreeNodePrograms({three: T, renderer: f.renderer}); t.after(() => owner.dispose());
  f.scene.updateMatrixWorld(); f.camera.updateMatrixWorld();
  const context = {scene: f.scene, camera: f.camera, object: f.object, material: f.material,
    lights: f.scene.children.filter(object => object.isLight), renderId: 1, frameId: 1};
  return {owner, context, compile: () => owner.compile(f.material, f.object, context)};
}

test('default MeshSSSNodeMaterial automatically submits the exact source physical fallback', async t => {
  const f = fixture(t); f.scene.add(new T.DirectionalLight());
  assert.equal(f.material.useSSS, false);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  source(f, captures(f)[0].binding.nodeProgram);
  assert.equal(native(f).length, 1); assert.equal(native(f)[0].args[0], f.object.geometry.index.count);
  assert.equal(native(f)[0].pipeline.depthStencil.depthWriteEnabled, true);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

for (const Light of [T.DirectionalLight, T.PointLight, T.SpotLight])
test(`SSS ${Light.name} scattering and thickness texture retain exact source WGSL and bindings`, async t => {
  const f = fixture(t), light = new Light(0xffffff, 2); light.position.set(1, 2, 3); f.scene.add(light);
  controls(f.material); const texture = map(t);
  f.material.thicknessColorNode = N.texture(texture).rgb.mul(f.material.thicknessColorNode);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const descriptor = captures(f)[0].binding.nodeProgram; source(f, descriptor);
  for (const slot of sssSlots) assert.match(descriptor.fragment, new RegExp(slot));
  const packet = values(captures(f)[0].nodeUniforms);
  for (const value of [.1875, .3125, .4375, .125, .375, .625, 2.75, 5.25]) assert.ok(packet.includes(value));
  const uploaded = f.writes.find(write => write.bytes.length === texture.image.data.length);
  assert.deepEqual(uploaded.bytes, texture.image.data);
  assert.ok(bindings(native(f)[0]).some(entry => entry.resource?.texture === uploaded.texture));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('SSS keeps physical clearcoat, sheen, iridescence, anisotropy and pinned retroreflection behavior', async t => {
  const f = fixture(t); f.scene.add(new T.PointLight()); controls(f.material);
  Object.assign(f.material, {clearcoat: .5, sheen: .25, iridescence: .375, anisotropy: .625, retroreflectivity: .75});
  f.material.clearcoatRoughnessNode = N.uniform(.1875); f.material.specularIntensityNode = N.uniform(.8125);
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const descriptor = captures(f)[0].binding.nodeProgram; source(f, descriptor);
  for (const term of ['clearcoat', 'sheen', 'iridescence', 'anisotropy']) assert.match(descriptor.fragment, new RegExp(term, 'i'));
  assert.equal(f.material.setupLightingModel().retroreflection, false);
  assert.doesNotMatch(descriptor.fragment, /BRDF_Retroreflection/);
  const before = descriptor;
  f.material.clearcoat = 0; f.material.clearcoatRoughnessNode = null;
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.notEqual(captures(f)[1].binding.nodeProgram, before); source(f, captures(f)[1].binding.nodeProgram);
});

test('queued SSS thickness controls preserve source FRAME, RENDER and OBJECT frequencies and prior bytes', async t => {
  const f = fixture(t), nodes = controls(f.material), calls = {frame: 0, render: 0, object: 0};
  f.scene.add(new T.PointLight());
  nodes[0].onFrameUpdate(() => { calls.frame++; });
  nodes[1].onRenderUpdate(() => { calls.render++; });
  nodes[2].onObjectUpdate(() => { calls.object++; });
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  const first = captures(f)[0], saved = first.nodeUniforms.map(bytes => bytes.slice());
  nodes[0].value.setRGB(.5625, .6875, .8125);
  for (let index = 1; index < nodes.length; index++) nodes[index].value += 8;
  f.material.roughness = .21875; f.renderer.render(f.scene, f.camera);
  const second = captures(f)[1];
  assert.equal(first.binding.nodeProgram, second.binding.nodeProgram);
  assert.deepEqual(calls, {frame: 1, render: 2, object: 2});
  for (const value of [.125, .375, .625, 2.75, 5.25]) {
    assert.ok(values(saved).includes(value)); assert.ok(values(second.nodeUniforms).includes(value + 8));
  }
  for (const node of nodes.slice(1)) node.value = 99;
  assert.deepEqual(first.nodeUniforms, saved); await drain(f.renderer);
  assert.equal(native(f).length, 2); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('each SSS slot replacement rejects stale capture before callbacks and rebuilds the source graph', async t => {
  const f = fixture(t); f.scene.add(new T.PointLight()); controls(f.material); let updates = 0;
  f.material.colorNode = N.uniform(new T.Color(.5, .25, .125)).onObjectUpdate(() => { updates++; });
  await f.renderer.init(); const {owner, context, compile} = programOwner(t, f);
  let descriptor = compile(); const first = owner.captureDraw(descriptor, context), saved = first.uniforms.map(bytes => bytes.slice());
  for (const [index, slot] of sssSlots.entries()) {
    const count = updates; f.material[slot] = index === 0 ? N.vec3(.75, .5, .25) : N.float(index + .25);
    assert.throws(() => owner.captureDraw(descriptor, context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
    assert.equal(updates, count); const next = compile(); assert.notEqual(next, descriptor);
    descriptor = next; source(f, descriptor); owner.captureDraw(descriptor, {...context, renderId: index + 2});
  }
  f.material.thicknessColorNode = null;
  assert.throws(() => owner.captureDraw(descriptor, context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const disabled = compile(); source(f, disabled); assert.equal(f.material.useSSS, false);
  assert.deepEqual(first.uniforms, saved);
});

for (const phase of ['construction', 'before', 'update'])
test(`SSS graph replacement during ${phase} rejects before native publication`, async t => {
  const f = fixture(t); f.scene.add(new T.PointLight()); controls(f.material); let calls = 0;
  const change = () => { calls++; f.material.thicknessPowerNode = N.float(7.5); return .125; };
  if (phase === 'construction') f.material.thicknessColorNode = N.Fn(() => { change(); return N.vec3(.5); })();
  else if (phase === 'before') {
    f.material.thicknessDistortionNode.updateBeforeType = 'object'; f.material.thicknessDistortionNode.updateBefore = change;
  } else f.material.thicknessDistortionNode.onObjectUpdate(change);
  await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(calls, 1); assert.equal(native(f).length, 0); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('inactive SSS terms keep their graph constructors and updates inert at source lighting boundaries', async t => {
  const f = fixture(t); let builds = 0, updates = 0;
  f.material.thicknessColorNode = N.Fn(() => { builds++; return N.vec3(.5); })();
  f.material.thicknessScaleNode = N.uniform(5).onObjectUpdate(() => { updates++; });
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  const ambient = new T.AmbientLight(); f.scene.add(ambient); f.renderer.render(f.scene, f.camera);
  f.scene.add(new T.PointLight()); f.renderer.lighting.enabled = false; f.renderer.render(f.scene, f.camera);
  f.renderer.lighting.enabled = true; f.material.thicknessColorNode = null; f.renderer.render(f.scene, f.camera);
  await drain(f.renderer);
  assert.equal(builds, 0); assert.equal(updates, 0); assert.equal(native(f).length, 4);
  f.material.thicknessColorNode = N.Fn(() => { builds++; return N.vec3(.5); })();
  f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  assert.equal(builds, 1); assert.equal(updates, 1); assert.equal(native(f).length, 5);
});

test('SSS material environment overrides the scene and removal restores its live source graph', async t => {
  const f = fixture(t); f.scene.add(new T.PointLight()); controls(f.material);
  const environment = new T.DataTexture(new Uint16Array(336 * 64 * 4).fill(0x3800), 336, 64, T.RGBAFormat, T.HalfFloatType);
  Object.assign(environment, {mapping: T.CubeUVReflectionMapping, minFilter: T.LinearFilter,
    magFilter: T.LinearFilter, colorSpace: T.LinearSRGBColorSpace}); environment.needsUpdate = true;
  t.after(() => environment.dispose());
  f.material.envMap = environment; f.material.envMapIntensity = .8125;
  let updates = 0;
  f.scene.environmentNode = N.uniform(new T.Color(.0625, .125, .25)).onObjectUpdate(() => { updates++; });
  await f.renderer.init(); f.renderer.render(f.scene, f.camera);
  const first = captures(f)[0], saved = first.nodeUniforms.map(bytes => bytes.slice()); source(f, first.binding.nodeProgram);
  assert.equal(updates, 0); assert.ok(values(saved).includes(.8125));
  f.material.envMap = null; f.renderer.render(f.scene, f.camera);
  const second = captures(f)[1]; source(f, second.binding.nodeProgram);
  assert.equal(updates, 1); assert.notEqual(first.binding.nodeProgram, second.binding.nodeProgram);
  assert.ok(values(second.nodeUniforms).includes(.0625)); assert.deepEqual(first.nodeUniforms, saved);
  await drain(f.renderer);
  const upload = f.writes.find(write => write.bytes.length === environment.image.data.byteLength);
  assert.ok(upload); assert.deepEqual(upload.bytes, new Uint8Array(environment.image.data.buffer));
  assert.ok(bindings(native(f)[0]).some(entry => entry.resource?.texture === upload.texture));
  assert.ok(!bindings(native(f)[1]).some(entry => entry.resource?.texture === upload.texture));
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('SSS transmission uses source opaque copies, inherited dispersion and guarded queued variants', async t => {
  const f = fixture(t); f.scene.add(new T.PointLight()); controls(f.material);
  Object.assign(f.material, {transmission: .75, thickness: .375, dispersion: .625, roughness: .25,
    attenuationDistance: 3.5, attenuationColor: new T.Color(.125, .5, .875)});
  const opaqueMaterial = new T.MeshBasicMaterial({color: 0x4488cc}), geometry = new T.PlaneGeometry();
  const opaque = new T.Mesh(geometry, opaqueMaterial); opaque.position.z = -1.5; f.scene.add(opaque);
  t.after(() => { opaqueMaterial.dispose(); geometry.dispose(); });
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const descriptor = captures(f)[0].binding.nodeProgram; source(f, descriptor);
  assert.match(descriptor.fragment, /getVolumeTransmissionRay/); assert.match(descriptor.fragment, /volumeAttenuation/);
  assert.ok(values(captures(f)[0].nodeUniforms).includes(.625)); assert.equal(f.copies.length, 1);
  assert.equal(f.copies[0].destination.texture.mipLevelCount, 6);
  const draw = native(f).find(draw => draw.pipeline.fragment.module.code.includes('thicknessScaleNode'));
  assert.ok(draw); assert.ok(bindings(draw).some(entry => entry.resource?.texture === f.copies[0].destination.texture));
  const {owner, context, compile} = programOwner(t, f); let previous = compile();
  for (const change of [
    () => { f.material.dispersion = 0; },
    () => { f.material.side = T.BackSide; },
    () => { f.material.thicknessNode = N.float(.875); },
    () => { f.material.transmission = 0; },
  ]) {
    const copyCount = f.copies.length; change();
    assert.throws(() => owner.captureDraw(previous, context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
    assert.equal(f.copies.length, copyCount); const next = compile(); assert.notEqual(next, previous);
    source(f, next); previous = next;
  }
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('SSS receives the native spot shadow while retaining ordinary caster and color draws', async t => {
  const f = fixture(t), light = new T.SpotLight(0xffffff, 8, 20); light.position.set(1, 2, 3);
  light.castShadow = true; light.shadow.mapSize.set(8, 8); f.scene.add(light); controls(f.material);
  f.renderer.shadowMap.enabled = true; f.object.receiveShadow = true;
  const material = new T.MeshBasicMaterial({color: 0x4488cc}), geometry = new T.BoxGeometry(.5, .5, .5);
  const blocker = new T.Mesh(geometry, material); blocker.castShadow = true; f.scene.add(blocker);
  t.after(() => { geometry.dispose(); material.dispose(); light.shadow.dispose(); });
  await f.renderer.init(); f.renderer.render(f.scene, f.camera); await drain(f.renderer);
  const draws = native(f), draw = draws.find(draw => draw.pipeline.fragment.module.code.includes('thicknessScaleNode'));
  assert.ok(draw); assert.ok(draws.length >= 3);
  const view = bindings(draw).map(entry => entry.resource).find(resource => resource?.texture?.format?.startsWith('depth'));
  assert.ok(view); assert.ok(f.device.passes.some(pass => pass.desc.depthStencilAttachment?.view.texture === view.texture));
  assert.match(draw.pipeline.fragment.module.code, /textureSampleCompare/); assert.equal(f.renderer._nodeCaptureBytes, 0);
});

test('SSS-generated storage writes are rejected before uploads and draw callbacks', async t => {
  const f = fixture(t); f.scene.add(new T.PointLight()); let updates = 0;
  f.material.thicknessScaleNode = N.uniform(5).onObjectUpdate(() => { updates++; });
  const storage = N.instancedArray(1, 'float');
  f.material.thicknessColorNode = N.Fn(() => { storage.element(0).assign(1); return N.vec3(.5); })();
  await f.renderer.init();
  assert.throws(() => f.renderer.render(f.scene, f.camera), {code: 'THREE_NODE_PROGRAM_STORAGE'});
  assert.equal(updates, 0); assert.equal(native(f).length, 0); assert.equal(f.writes.length, 0);
  assert.equal(f.renderer._nodeCaptureBytes, 0);
});
