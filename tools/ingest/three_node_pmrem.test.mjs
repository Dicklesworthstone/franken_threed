/** Authored source PMREM graphs retain real atlas bindings and source setup
 * identity. These builder tests assert admission and capture, not GPU pixels. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
const N = THREE.TSL;
function cubeUV(value = 0x3800) {
  const texture = new THREE.DataTexture(new Uint16Array(336 * 64 * 4).fill(value), 336, 64,
    THREE.RGBAFormat, THREE.HalfFloatType);
  texture.mapping = THREE.CubeUVReflectionMapping;
  texture.minFilter = texture.magFilter = THREE.LinearFilter;
  texture.colorSpace = THREE.LinearSRGBColorSpace; texture.needsUpdate = true;
  return texture;
}
function fixture(t) {
  const renderer = {library: new THREE.StandardNodeLibrary(), contextNode: N.context(),
    debug: {diagnostics: {keywords: false}}, depth: true, lighting: {enabled: true},
    backend: {hasFeature: () => false, utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})}},
    hasFeature: () => false, getRenderTarget: () => null, getMRT: () => null, currentSamples: 1,
    _nodes: {nodeFrame: new THREE.NodeFrame()}, info: {calls: 1}};
  const programs = createThreeNodePrograms({three: THREE, renderer}); t.after(() => programs.dispose());
  const material = new THREE.NodeMaterial(); material.lights = false;
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(50, 1, .1, 30);
  scene.backgroundBlurriness = .375; scene.backgroundIntensity = .625;
  const pmrem = N.pmremTexture(cubeUV()), multiplier = N.uniform(1);
  material.colorNode = N.vec4(pmrem).mul(N.backgroundIntensity).mul(multiplier).context({
    getUV: () => N.backgroundRotation.mul(N.normalWorldGeometry),
    getTextureLevel: () => N.backgroundBlurriness,
  });
  const object = new THREE.Mesh(new THREE.SphereGeometry(1, 8, 6), material); scene.add(object);
  camera.position.z = 4; camera.coordinateSystem = THREE.WebGPUCoordinateSystem; camera.updateProjectionMatrix();
  scene.updateMatrixWorld(); camera.updateMatrixWorld();
  const context = {scene, camera, object, material, renderId: 1, frameId: 1};
  return {programs, material, object, scene, camera, pmrem, multiplier, context,
    compile: () => programs.compile(material, object, context)};
}
const values = packet => packet.uniforms.flatMap(bytes => [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);
const sampled = packet => packet.textures.filter(binding => binding.isSampledTexture).map(binding => binding.texture);

test('authored PMREM nodes use the exact CubeUV graph and only their resolved atlas binding', t => {
  const f = fixture(t), descriptor = f.compile(), first = f.programs.captureDraw(descriptor, f.context);
  assert.match(descriptor.fragment, /roughnessToMip/); assert.match(descriptor.fragment, /getFace/);
  assert.deepEqual(sampled(first), [f.pmrem.value]);
  assert.ok(values(first).includes(.375)); assert.ok(values(first).includes(.625));
  const saved = first.uniforms.map(bytes => bytes.slice());
  f.scene.backgroundBlurriness = .75; f.scene.backgroundIntensity = .875; f.scene.backgroundRotation.y = Math.PI / 2;
  assert.equal(f.compile(), descriptor);
  const second = f.programs.captureDraw(descriptor, {...f.context, renderId: 2});
  assert.ok(values(second).includes(.75)); assert.ok(values(second).includes(.875));
  assert.notDeepEqual(first.uniforms, second.uniforms); assert.deepEqual(first.uniforms, saved);
});
test('same-format PMREM input replacement rebuilds setup and rejects a stale capture before callbacks', t => {
  const f = fixture(t); let calls = 0;
  f.multiplier.onObjectUpdate(() => { calls++; return 1; });
  const initial = f.pmrem.value, a = f.compile(), old = f.programs.captureDraw(a, f.context);
  f.pmrem.value = cubeUV(0x3400);
  assert.throws(() => f.programs.captureDraw(a, {...f.context, renderId: 2}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(calls, 1);
  const b = f.compile(); assert.notEqual(a, b);
  const next = f.programs.captureDraw(b, {...f.context, renderId: 2});
  assert.equal(calls, 2); assert.deepEqual(sampled(old), [initial]); assert.deepEqual(sampled(next), [f.pmrem.value]);
  assert.equal(f.programs.diagnostics.programs, 1);
});
test('PMREM render-target orientation changes rebuild the source sampling expression', t => {
  const f = fixture(t), a = f.compile();
  f.pmrem.value.isRenderTargetTexture = true;
  assert.throws(() => f.programs.captureDraw(a, f.context), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const b = f.compile(); assert.notEqual(a, b); assert.notEqual(a.fragment, b.fragment);
});
test('direct CubeUV pixel and PMREM versions keep an authored PMREM graph reusable', t => {
  const f = fixture(t), descriptor = f.compile();
  f.programs.captureDraw(descriptor, f.context);
  f.pmrem.value.image.data.fill(0x3400); f.pmrem.value.needsUpdate = true; f.pmrem.value.needsPMREMUpdate = true;
  assert.equal(f.compile(), descriptor);
  assert.deepEqual(sampled(f.programs.captureDraw(descriptor, {...f.context, renderId: 2})), [f.pmrem.value]);
});
test('unsupported authored PMREM sources and custom subclasses refuse before callbacks', t => {
  const f = fixture(t); let calls = 0;
  f.multiplier.onObjectUpdate(() => { calls++; return 1; });
  f.pmrem.value = new THREE.CubeTexture();
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_ENVIRONMENT'});
  const storage = new THREE.StorageTexture(16, 16); storage.mapping = THREE.CubeUVReflectionMapping;
  f.pmrem.value = storage; assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_TEXTURE'});
  class CustomPMREM extends THREE.PMREMNode {}
  f.material.colorNode = N.vec4(new CustomPMREM(cubeUV(), N.vec3(0, 0, 1), N.float(.5)));
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_TEXTURE'});
  assert.equal(calls, 0);
});
