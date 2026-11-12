/** Pinned IES/projector graphs and their source FRAME update semantics. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';

const N = T.TSL;
function map(t, value = .5, {width = 4, height = 1} = {}) {
  const texture = new T.DataTexture(new Uint16Array(width * height).fill(T.DataUtils.toHalfFloat(value)),
    width, height, T.RedFormat, T.HalfFloatType);
  texture.minFilter = texture.magFilter = T.LinearFilter;
  texture.needsUpdate = true; t.after(() => texture.dispose()); return texture;
}
function fixture(t, Light, Material = T.MeshLambertMaterial) {
  const frame = new T.NodeFrame(), renderer = {library: new T.StandardNodeLibrary(), contextNode: N.context(),
    debug: {diagnostics: {keywords: false}}, depth: true, lighting: {enabled: true}, shadowMap: {enabled: false},
    backend: {hasFeature: () => false, utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})}},
    hasFeature: () => false, getRenderTarget: () => null, getMRT: () => null, currentSamples: 1,
    _nodes: {nodeFrame: frame}, info: {calls: 1}};
  const owner = createThreeNodePrograms({three: T, renderer}); t.after(() => owner.dispose());
  const material = new Material(), object = new T.Mesh(new T.PlaneGeometry(), material);
  const light = new Light(0xffffff, 2); light.position.set(1, 2, 3);
  const scene = new T.Scene(); scene.add(object, light, light.target); scene.updateMatrixWorld();
  const camera = new T.PerspectiveCamera(50, 1, .1, 20); camera.position.z = 4;
  camera.coordinateSystem = T.WebGPUCoordinateSystem; camera.updateProjectionMatrix(); camera.updateMatrixWorld();
  const context = {object, material, scene, camera, lights: [light], frameId: 1, renderId: 1};
  t.after(() => { object.geometry.dispose(); material.dispose(); light.dispose(); });
  return {owner, renderer, frame, material, object, light, scene, camera, context,
    compile: () => owner.compile(material, object, context),
    capture: (descriptor, overrides = {}) => owner.captureDraw(descriptor, {...context, ...overrides})};
}
const sampled = packet => packet.textures.filter(binding => binding.isSampledTexture).map(binding => binding.texture);
const floats = packet => packet.uniforms.flatMap(bytes => [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);
function assertSource(f, descriptor) {
  const source = new T.WGSLNodeBuilder(f.object, f.renderer);
  source.scene = f.scene; source.camera = f.camera; source.lightsNode = N.lights([f.light]); source.build();
  assert.equal(descriptor.vertex, source.vertexShader); assert.equal(descriptor.fragment, source.fragmentShader);
}

for (const Light of [T.IESSpotLight, T.ProjectorLight])
for (const Material of [T.MeshLambertMaterial, T.MeshPhongNodeMaterial, T.MeshPhysicalMaterial])
test(`${Light.name} with ${Material.name} retains the complete source attenuation shader`, t => {
  const f = fixture(t, Light, Material), texture = map(t);
  f.light[Light === T.IESSpotLight ? 'iesMap' : 'map'] = texture;
  const descriptor = f.compile(); assertSource(f, descriptor);
  assert.match(descriptor.fragment, /textureSample/);
  assert.ok(sampled(f.capture(descriptor)).includes(texture));
});

for (const Light of [T.IESSpotLight, T.ProjectorLight])
test(`${Light.name} retains the inherited projected color Fn and ignores its shadowed map slot`, t => {
  const f = fixture(t, Light), gain = N.uniform(.375).setName('spotFamilyGain');
  if (Light === T.IESSpotLight) f.light.iesMap = map(t);
  else f.light.aspect = 1.25;
  f.light.map = {ignored: true};
  f.light.colorNode = N.Fn(([uv]) => N.vec3(uv.xy, .5).mul(gain));
  const descriptor = f.compile(); assertSource(f, descriptor);
  assert.ok(floats(f.capture(descriptor)).includes(.375));
  gain.value = .8125;
  assert.equal(f.compile(), descriptor);
  assert.ok(floats(f.capture(descriptor, {renderId: 2})).includes(.8125));
});

test('IES replacement maps follow source FRAME frequency and retain the constructed attenuation branch', t => {
  const f = fixture(t, T.IESSpotLight), first = map(t, .25), second = map(t, .75);
  f.light.iesMap = first;
  const descriptor = f.compile(), accepted = f.capture(descriptor);
  assert.deepEqual(sampled(accepted), [first]);
  f.light.iesMap = second;
  assert.equal(f.compile(), descriptor, 'IES replacement is a source light update, not graph invalidation');
  assert.deepEqual(sampled(f.capture(descriptor, {renderId: 2})), [first]);
  assert.deepEqual(sampled(f.capture(descriptor, {frameId: 2, renderId: 3})), [second]);
  assert.deepEqual(sampled(accepted), [first], 'accepted texture identities stay immutable');
  f.light.iesMap = null;
  assert.equal(f.compile(), descriptor);
  assert.deepEqual(sampled(f.capture(descriptor, {frameId: 3, renderId: 4})), [second],
    'source IES update leaves its last texture installed when the public map is null');
  f.material.needsUpdate = true;
  const cone = f.compile(); assert.notEqual(cone, descriptor);
  assert.deepEqual(sampled(f.capture(cone, {frameId: 4, renderId: 5})), []);
  f.light.iesMap = first;
  assert.equal(f.compile(), cone, 'adding IES data cannot rewrite an already constructed ordinary cone');
  assert.deepEqual(sampled(f.capture(cone, {frameId: 5, renderId: 6})), []);
  f.material.needsUpdate = true;
  const restored = f.compile();
  assert.deepEqual(sampled(f.capture(restored, {frameId: 6, renderId: 7})), [first]);
});

test('projector aspect and penumbra retain the actual source per-frame update', t => {
  const f = fixture(t, T.ProjectorLight), source = new T.ProjectorLight(), sourceNode = new T.ProjectorLightNode(source);
  f.light.map = source.map = map(t, .5, {width: 4, height: 2});
  f.light.penumbra = source.penumbra = 1;
  const descriptor = f.compile(), first = f.capture(descriptor); sourceNode.update();
  assert.equal(f.light.shadow.aspect, source.shadow.aspect);
  assert.ok(floats(first).includes(Math.fround(sourceNode.penumbraCosNode.value)));
  f.light.aspect = source.aspect = .75;
  f.capture(descriptor, {renderId: 2});
  assert.equal(f.light.shadow.aspect, 2, 'same-frame draws retain the previous source aspect update');
  sourceNode.update();
  f.capture(descriptor, {frameId: 2, renderId: 3});
  assert.equal(f.light.shadow.aspect, source.shadow.aspect); assert.equal(f.compile(), descriptor);
});

for (const Light of [T.IESSpotLight, T.ProjectorLight])
test(`${Light.name} registrations and inherited subclasses retain exact source shaders and warm caches`, t => {
  const f = fixture(t, Light);
  f.renderer.library.lightNodes.set(Light, T.SpotLightNode);
  const replacement = f.compile(); assertSource(f, replacement);
  assert.equal(f.compile(), replacement, 'a source-valid builtin replacement retains its warm program');
  assert.doesNotThrow(() => f.capture(replacement));
  class CustomLight extends Light {}
  class CustomNode extends T[`${Light.name}Node`] {}
  const custom = new CustomLight(), texture = map(t);
  t.after(() => custom.dispose());
  custom[Light === T.IESSpotLight ? 'iesMap' : 'map'] = texture;
  f.renderer.library.addLight(CustomNode, CustomLight);
  f.scene.add(custom, custom.target); f.scene.updateMatrixWorld();
  const context = {...f.context, lights: [custom]}, descriptor = f.owner.compile(f.material, f.object, context);
  assertSource({...f, light: custom}, descriptor);
  assert.equal(f.owner.compile(f.material, f.object, context), descriptor);
  assert.ok(sampled(f.owner.captureDraw(descriptor, context)).includes(texture));
});
