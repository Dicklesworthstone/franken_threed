/** Material-local PBR environments retain the pinned source graph and updates.
 * These checks execute source construction/capture, not GPU pixel parity. */
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
  const material = new Material({metalness: .25, roughness: .5, envMap: cubeUV(), envMapIntensity: .375});
  material.colorNode = N.vec3(.5, .25, .125);
  const object = new THREE.Mesh(new THREE.SphereGeometry(1, 8, 6), material), scene = new THREE.Scene();
  scene.add(object); scene.environment = cubeUV(0x3400); scene.environmentIntensity = .0625;
  const camera = new THREE.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  camera.coordinateSystem = THREE.WebGPUCoordinateSystem; camera.updateProjectionMatrix();
  scene.updateMatrixWorld(); camera.updateMatrixWorld();
  const context = {scene, camera, lights: [], object, material, renderId: 1, frameId: 1};
  return {renderer, programs, material, object, scene, camera, context,
    compile: () => programs.compile(material, object, context)};
}
const sampled = packet => packet.textures.filter(binding => binding.isSampledTexture).map(binding => binding.texture);
const values = packet => packet.uniforms.flatMap(bytes => [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);
function sourceBuilder(f) {
  const builder = new THREE.WGSLNodeBuilder(f.object, f.renderer);
  builder.material = f.material; builder.context.material = f.material;
  builder.scene = f.scene; builder.camera = f.camera;
  builder.lightsNode = N.lights([]); builder.environmentNode = N.texture(f.scene.environment);
  builder.build(); return builder;
}

for (const Material of [THREE.MeshStandardMaterial, THREE.MeshPhysicalMaterial,
  THREE.MeshStandardNodeMaterial, THREE.MeshPhysicalNodeMaterial]) {
  test(`${Material.name} material envMap overrides the scene with byte-identical source WGSL`, t => {
    const f = fixture(t, Material);
    if (f.material.isMeshPhysicalMaterial || f.material.isMeshPhysicalNodeMaterial) {
      f.material.clearcoat = .5; f.material.sheen = .25; f.material.iridescence = .375;
    }
    const reference = sourceBuilder(f), descriptor = f.compile();
    assert.equal(descriptor.vertex, reference.vertexShader);
    assert.equal(descriptor.fragment, reference.fragmentShader);
    const packet = f.programs.captureDraw(descriptor, f.context), textures = sampled(packet);
    assert.ok(textures.includes(f.material.envMap)); assert.ok(!textures.includes(f.scene.environment));
    assert.equal(textures.length, 2); assert.ok(textures.some(texture => texture.name === 'DFG_LUT'));
    assert.ok(values(packet).includes(.375)); assert.ok(!values(packet).includes(.0625));
    assert.match(descriptor.fragment, /iblIrradiance/); assert.match(descriptor.fragment, /roughnessToMip/);
  });
}

test('material intensity and rotation update per draw without recompilation or rewriting accepted bytes', t => {
  const f = fixture(t), descriptor = f.compile(), first = f.programs.captureDraw(descriptor, f.context);
  const saved = first.uniforms.map(bytes => bytes.slice());
  f.material.envMapIntensity = .875; f.material.envMapRotation.set(.2, .7, -.3);
  f.scene.environmentIntensity = .125; f.scene.environmentRotation.y = Math.PI;
  assert.equal(f.compile(), descriptor);
  const second = f.programs.captureDraw(descriptor, {...f.context, renderId: 2});
  assert.ok(values(first).includes(.375)); assert.ok(values(second).includes(.875));
  assert.ok(!values(second).includes(.125));
  const expected = new THREE.Matrix4().makeRotationFromEuler(f.material.envMapRotation).transpose().elements;
  const data = values(second), rotation = expected.map(Math.fround);
  assert.ok(data.some((_, i) => rotation.every((value, j) => data[i + j] === value)), 'the source inverse material rotation is captured');
  assert.notDeepEqual(first.uniforms, second.uniforms); assert.deepEqual(first.uniforms, saved);
});

test('material environment replacement, scene fallback and removal preserve prior texture captures', t => {
  const f = fixture(t), initial = f.material.envMap, a = f.compile();
  const first = f.programs.captureDraw(a, f.context);
  let calls = 0;
  f.material.colorNode = N.uniform(new THREE.Color(.5, .25, .125)).onObjectUpdate(() => {
    calls++; return new THREE.Color(.5, .25, .125);
  });
  const withUpdates = f.compile(); f.programs.captureDraw(withUpdates, {...f.context, renderId: 2});
  f.material.envMap = cubeUV(0x3000);
  assert.throws(() => f.programs.captureDraw(withUpdates, {...f.context, renderId: 3}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(calls, 1, 'stale capture rejects before authored callbacks');
  const b = f.compile(), second = f.programs.captureDraw(b, {...f.context, renderId: 3});
  assert.ok(sampled(first).includes(initial)); assert.ok(sampled(second).includes(f.material.envMap));
  f.material.envMap = null;
  const c = f.compile(), fallback = f.programs.captureDraw(c, {...f.context, renderId: 4});
  assert.notEqual(c, b); assert.ok(sampled(fallback).includes(f.scene.environment));
  assert.ok(values(fallback).includes(.0625));
  f.scene.environment = null;
  const d = f.compile(), removed = f.programs.captureDraw(d, {...f.context, renderId: 5});
  assert.notEqual(d, c); assert.equal(sampled(removed).length, 0);
  assert.equal(f.programs.diagnostics.programs, 1);
});

test('a material override leaves an unused custom scene environment inert', t => {
  const f = fixture(t); f.scene.environmentNode = N.vec3(.25, .5, 1);
  const descriptor = f.compile();
  assert.ok(sampled(f.programs.captureDraw(descriptor, f.context)).includes(f.material.envMap));
  f.material.envMap = null;
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_ENVIRONMENT'});
});

test('sharing the same map between scene and material preserves source ownership transitions', t => {
  const f = fixture(t); f.scene.environment = f.material.envMap;
  let calls = 0;
  f.material.colorNode = N.uniform(new THREE.Color(.5, .25, .125)).onObjectUpdate(() => {
    calls++; return new THREE.Color(.5, .25, .125);
  });
  const a = f.compile(), material = f.programs.captureDraw(a, f.context);
  f.material.envMap = null;
  assert.throws(() => f.programs.captureDraw(a, {...f.context, renderId: 2}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(calls, 1, 'same-image material-to-scene changes reject before authored callbacks');
  const b = f.compile(), scene = f.programs.captureDraw(b, {...f.context, renderId: 2});
  assert.notEqual(a, b); assert.ok(values(material).includes(.375)); assert.ok(values(scene).includes(.0625));
  assert.deepEqual(sampled(material), sampled(scene));
  f.material.envMap = f.scene.environment;
  assert.throws(() => f.programs.captureDraw(b, {...f.context, renderId: 3}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(calls, 2, 'same-image scene-to-material changes reject before authored callbacks');
  const c = f.compile(); assert.notEqual(c, b);
  assert.ok(values(f.programs.captureDraw(c, {...f.context, renderId: 3})).includes(.375));
});

test('custom environment lighting cannot enter through authored graphics slots', t => {
  const f = fixture(t);
  f.material.colorNode = new THREE.EnvironmentNode(N.texture(f.material.envMap));
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_LIGHTS'});
});

test('unused material environment inputs remain inert for disabled lighting and authored fragments', t => {
  const f = fixture(t); f.material.envMap = new THREE.Texture();
  f.material.lights = false;
  assert.equal(sampled(f.programs.captureDraw(f.compile(), f.context)).length, 0);
  f.material.lights = true; f.renderer.lighting.enabled = false;
  assert.equal(sampled(f.programs.captureDraw(f.compile(), {...f.context, renderId: 2})).length, 0);
  f.renderer.lighting.enabled = true; f.material.fragmentNode = N.vec4(1, 0, 0, 1);
  assert.equal(sampled(f.programs.captureDraw(f.compile(), {...f.context, renderId: 3})).length, 0);
  f.material.fragmentNode = null;
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_ENVIRONMENT'});
});
