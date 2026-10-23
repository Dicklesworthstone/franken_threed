/** Source PBR environment graph construction and per-draw values. GPU/source
 * parity is verified separately; these tests execute the real pinned builder. */
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
function fixture(t, Material = THREE.MeshStandardNodeMaterial) {
  const renderer = {library: new THREE.StandardNodeLibrary(), contextNode: N.context(),
    debug: {diagnostics: {keywords: false}}, depth: true, lighting: {enabled: true},
    backend: {hasFeature: () => false, utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})}},
    hasFeature: () => false, getRenderTarget: () => null, getMRT: () => null, currentSamples: 1,
    _nodes: {nodeFrame: new THREE.NodeFrame()}, info: {calls: 1}};
  const programs = createThreeNodePrograms({three: THREE, renderer}); t.after(() => programs.dispose());
  const material = new Material({metalness: .25, roughness: .5}); material.colorNode = N.vec3(.5, .25, .125);
  const object = new THREE.Mesh(new THREE.SphereGeometry(1, 8, 6), material), scene = new THREE.Scene();
  scene.add(object); scene.environment = cubeUV();
  const camera = new THREE.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  camera.coordinateSystem = THREE.WebGPUCoordinateSystem; camera.updateProjectionMatrix();
  scene.updateMatrixWorld(); camera.updateMatrixWorld();
  const context = {scene, camera, lights: [], object, material, renderId: 1, frameId: 1};
  return {renderer, programs, material, object, scene, camera, context,
    compile: () => programs.compile(material, object, context)};
}
const values = capture => capture.uniforms.flatMap(bytes => Array.from(new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)));
for (const Material of [THREE.MeshStandardNodeMaterial, THREE.MeshPhysicalNodeMaterial]) {
  test(`${Material.name} uses source PMREM diffuse/specular lighting without analytic lights`, t => {
    const f = fixture(t, Material);
    if (f.material.isMeshPhysicalNodeMaterial) { f.material.clearcoat = .5; f.material.sheen = .25; }
    const descriptor = f.compile(), packet = f.programs.captureDraw(descriptor, f.context);
    assert.match(descriptor.fragment, /iblIrradiance/); assert.match(descriptor.fragment, /indirectSpecular/);
    assert.match(descriptor.fragment, /getFace/); assert.match(descriptor.fragment, /roughnessToMip/);
    const resources = packet.textures.filter(binding => binding.isSampledTexture).map(binding => binding.texture);
    assert.deepEqual(new Set(resources), new Set([f.scene.environment, resources.find(texture => texture.name === 'DFG_LUT')]));
    assert.ok(resources.some(texture => texture.name === 'DFG_LUT'));
    assert.equal(resources.filter(texture => texture === f.scene.environment).length, 1);
    assert.equal(resources.length, 2, 'unbound source function texture placeholders never become GPU resources');
    if (f.material.isMeshPhysicalNodeMaterial) assert.match(descriptor.fragment, /clearcoatRadiance/);
  });
}
test('environment rotation and intensity remain call-time uniforms while compatible draws reuse their graph', t => {
  const f = fixture(t); f.scene.environmentIntensity = .375;
  const descriptor = f.compile(), before = f.programs.captureDraw(descriptor, f.context);
  const saved = before.uniforms.map(bytes => bytes.slice());
  f.scene.environmentIntensity = .875; f.scene.environmentRotation.y = Math.PI / 2;
  assert.equal(f.compile(), descriptor);
  const after = f.programs.captureDraw(descriptor, {...f.context, renderId: 2});
  assert.ok(values(before).includes(.375)); assert.ok(values(after).includes(.875));
  assert.notDeepEqual(after.uniforms, before.uniforms); assert.deepEqual(before.uniforms, saved);
  assert.equal(before.textures.find(binding => binding.texture === f.scene.environment).texture, f.scene.environment);
});
test('environment identity and removal select fresh graphs without rewriting earlier captured bindings', t => {
  const f = fixture(t), initial = f.scene.environment;
  const a = f.compile(), old = f.programs.captureDraw(a, f.context);
  f.scene.environment = cubeUV(0x3400);
  const b = f.compile(), replacement = f.programs.captureDraw(b, {...f.context, renderId: 2});
  assert.notEqual(a, b); assert.ok(replacement.textures.some(binding => binding.texture === f.scene.environment));
  assert.ok(old.textures.some(binding => binding.texture === initial));
  f.scene.environment = null;
  const c = f.compile(), removed = f.programs.captureDraw(c, {...f.context, renderId: 3});
  assert.notEqual(b, c); assert.doesNotMatch(c.fragment, /fn getFace/);
  assert.equal(removed.textures.filter(binding => binding.isSampledTexture).length, 0, 'the source skips lighting and DFG when no lights or environment remain');
  assert.equal(f.programs.diagnostics.programs, 1);
});
test('unsupported environment inputs refuse before authored node updates; source-disabled lighting stays inert', t => {
  const f = fixture(t); let calls = 0;
  f.material.colorNode = N.uniform(new THREE.Color()).onObjectUpdate(() => { calls++; return new THREE.Color(1, 0, 0); });
  f.scene.environmentNode = N.vec3(1);
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_ENVIRONMENT'}); assert.equal(calls, 0);
  f.scene.environmentNode = null; f.scene.environment = new THREE.Texture();
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_ENVIRONMENT'}); assert.equal(calls, 0);
  f.material.lights = false; assert.doesNotThrow(f.compile); assert.equal(calls, 0);
  f.material.lights = true; f.renderer.lighting.enabled = false;
  assert.doesNotThrow(f.compile); assert.equal(calls, 0);
});

test('direct CubeUV source updates keep ordinary texture versions without regeneration restrictions', t => {
  const f = fixture(t), environment = f.scene.environment; let calls = 0;
  f.material.colorNode = N.uniform(new THREE.Color(.5, .25, .125)).onObjectUpdate(() => { calls++; return new THREE.Color(.5, .25, .125); });
  const descriptor = f.compile();
  f.programs.captureDraw(descriptor, f.context);
  environment.image.data.fill(0x3400); environment.needsUpdate = true; environment.needsPMREMUpdate = true;
  assert.equal(f.compile(), descriptor);
  const captured = f.programs.captureDraw(descriptor, {...f.context, renderId: 2});
  assert.ok(captured.textures.some(binding => binding.texture === environment)); assert.equal(calls, 2);
});

test('same-identity environment orientation and source PMREM flags rebuild source setup', t => {
  const f = fixture(t), a = f.compile();
  f.scene.environment.isRenderTargetTexture = true;
  const b = f.compile(); assert.notEqual(a, b); assert.notEqual(a.fragment, b.fragment);
  f.scene.environment.isPMREMTexture = true;
  const c = f.compile(); assert.notEqual(b, c);
  f.scene.environment.needsUpdate = true; f.scene.environment.needsPMREMUpdate = true;
  assert.equal(f.compile(), c, 'direct CubeUV versions remain ordinary uploads rather than structural cache dimensions');
});
