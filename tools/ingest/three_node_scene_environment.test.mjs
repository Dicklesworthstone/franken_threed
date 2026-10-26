/** Scene environment graphs execute the exact r186 graph and update lifecycle.
 * This is source construction/capture evidence, not browser pixel parity. */
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
  const material = new Material({metalness: .25, roughness: .5, envMap: null, envMapIntensity: .375});
  material.color.setRGB(.5, .25, .125);
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
function assertSourceGraph(f, descriptor) {
  const builder = new THREE.WGSLNodeBuilder(f.object, f.renderer);
  builder.material = f.material; builder.context.material = f.material;
  builder.scene = f.scene; builder.camera = f.camera;
  builder.lightsNode = N.lights([]);
  builder.environmentNode = f.scene.environmentNode?.isNode ? f.scene.environmentNode
    : f.scene.environment ? N.texture(f.scene.environment) : null;
  builder.build();
  assert.equal(descriptor.vertex, builder.vertexShader); assert.equal(descriptor.fragment, builder.fragmentShader);
}

for (const Material of [THREE.MeshStandardMaterial, THREE.MeshPhysicalMaterial,
  THREE.MeshStandardNodeMaterial, THREE.MeshPhysicalNodeMaterial]) {
  test(`${Material.name} selects procedural scene environments with exact source WGSL`, t => {
    const f = fixture(t, Material);
    f.scene.environmentNode = N.vec3(.125, .25, .5);
    f.scene.environment = new THREE.Texture();
    if (f.material.isMeshPhysicalMaterial || f.material.isMeshPhysicalNodeMaterial) {
      f.material.clearcoat = .5; f.material.sheen = .25; f.material.iridescence = .375;
    }
    const descriptor = f.compile(); assertSourceGraph(f, descriptor);
    const packet = f.programs.captureDraw(descriptor, f.context);
    assert.deepEqual(sampled(packet).map(texture => texture.name), ['DFG_LUT']);
    assert.match(descriptor.fragment, /iblIrradiance/); assert.doesNotMatch(descriptor.fragment, /fn getFace/);
    assert.ok(values(packet).includes(.0625), 'the scene intensity accessor remains active');
  });
}

test('scene PMREM expressions inherit source roughness, rotation and physical radiance contexts', t => {
  const f = fixture(t, THREE.MeshPhysicalNodeMaterial), atlas = cubeUV(0x3000);
  f.scene.environmentNode = N.pmremTexture(atlas); f.material.clearcoat = .5; f.material.anisotropy = .25;
  f.scene.environmentRotation.set(.2, .6, -.4);
  const descriptor = f.compile(); assertSourceGraph(f, descriptor);
  const packet = f.programs.captureDraw(descriptor, f.context);
  assert.ok(sampled(packet).includes(atlas)); assert.ok(!sampled(packet).includes(f.scene.environment));
  assert.match(descriptor.fragment, /roughnessToMip/); assert.match(descriptor.fragment, /clearcoatRadiance/);
  const rotation = new THREE.Matrix4().makeRotationFromEuler(f.scene.environmentRotation).transpose().elements.map(Math.fround);
  const data = values(packet);
  assert.ok(data.some((_, i) => rotation.every((value, j) => data[i + j] === value)));
});

test('direct scene texture nodes rebuild source PMREM setup on compatible input replacement', t => {
  const f = fixture(t), initial = cubeUV(), input = N.texture(initial);
  f.scene.environmentNode = input; let calls = 0;
  f.material.colorNode = N.uniform(new THREE.Color(.5, .25, .125)).onObjectUpdate(() => {
    calls++; return new THREE.Color(.5, .25, .125);
  });
  const a = f.compile(), first = f.programs.captureDraw(a, f.context); assertSourceGraph(f, a);
  input.value = cubeUV(0x3000);
  assert.throws(() => f.programs.captureDraw(a, {...f.context, renderId: 2}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(calls, 1);
  const b = f.compile(), second = f.programs.captureDraw(b, {...f.context, renderId: 2}); assertSourceGraph(f, b);
  assert.notEqual(a, b); assert.ok(sampled(first).includes(initial)); assert.ok(sampled(second).includes(input.value));
  assert.ok(!sampled(second).includes(initial)); assert.equal(calls, 2);
});

test('scene graph replacement, structural versions and removal invalidate before observable updates', t => {
  const f = fixture(t); let calls = 0;
  f.material.colorNode = N.uniform(new THREE.Color()).onObjectUpdate(() => { calls++; return new THREE.Color(.5, .25, .125); });
  f.scene.environmentNode = N.vec3(.125, .25, .5);
  const a = f.compile(); f.programs.captureDraw(a, f.context);
  f.scene.environmentNode = N.vec3(N.float(.5), N.float(.25), N.float(.125));
  assert.throws(() => f.programs.captureDraw(a, {...f.context, renderId: 2}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(calls, 1);
  const b = f.compile(); f.programs.captureDraw(b, {...f.context, renderId: 2}); assert.notEqual(a, b);
  f.scene.environmentNode.node.nodes[0] = N.float(.75);
  f.scene.environmentNode.needsUpdate = true;
  assert.throws(() => f.programs.captureDraw(b, {...f.context, renderId: 3}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const c = f.compile(); assert.notEqual(c, b);
  f.programs.captureDraw(c, {...f.context, renderId: 3});
  f.scene.environmentNode = null;
  assert.throws(() => f.programs.captureDraw(c, {...f.context, renderId: 4}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(calls, 3);
  const d = f.compile(); assert.notEqual(c, d);
  assert.ok(sampled(f.programs.captureDraw(d, {...f.context, renderId: 4})).includes(f.scene.environment));
});

test('shared scene graph updates retain source FRAME, RENDER and OBJECT frequencies and context', t => {
  const f = fixture(t), calls = {frame: 0, render: 0, object: 0}, contexts = [];
  const frame = N.uniform(.125).onFrameUpdate(context => { calls.frame++; contexts.push(context.scene); return .125; });
  const render = N.uniform(.25).onRenderUpdate(context => { calls.render++; contexts.push(context.scene); return context.scene.userData.irradiance; });
  const object = N.uniform(.5).onObjectUpdate(context => { calls.object++; contexts.push(context.scene); return context.object.userData.irradiance; });
  f.scene.environmentNode = N.vec3(frame, render, object); f.scene.userData.irradiance = .3125;
  f.object.userData.irradiance = .4375;
  const other = new THREE.Mesh(f.object.geometry, f.material); other.userData.irradiance = .9375;
  f.scene.add(other); f.scene.updateMatrixWorld();
  const a = f.compile(), otherContext = {...f.context, object: other};
  const b = f.programs.compile(f.material, other, otherContext);
  const first = f.programs.captureDraw(a, f.context), saved = first.uniforms.map(bytes => bytes.slice());
  const second = f.programs.captureDraw(b, otherContext);
  assert.deepEqual(calls, {frame: 1, render: 1, object: 2});
  assert.ok(values(first).includes(.4375)); assert.ok(values(second).includes(.9375));
  f.scene.userData.irradiance = .6875;
  f.programs.captureDraw(a, {...f.context, renderId: 2});
  assert.deepEqual(calls, {frame: 1, render: 2, object: 3}); assert.equal(f.compile(), a);
  f.programs.captureDraw(b, {...otherContext, renderId: 3, frameId: 2});
  assert.deepEqual(calls, {frame: 2, render: 3, object: 4});
  assert.ok(contexts.every(scene => scene === f.scene)); assert.deepEqual(first.uniforms, saved);
});

test('material envNode and envMap retain precedence over invalid or replaced scene graphs', t => {
  const f = fixture(t);
  f.scene.environmentNode = new THREE.EnvironmentNode(N.vec3(1));
  f.material.envMap = cubeUV(); const map = f.material.envMap;
  const a = f.compile(); assertSourceGraph(f, a);
  assert.ok(sampled(f.programs.captureDraw(a, f.context)).includes(map));
  f.scene.environmentNode = N.vec3(.25, .5, .75); assert.equal(f.compile(), a);
  f.material.envNode = N.vec3(.5, .25, .125);
  const b = f.compile(); assertSourceGraph(f, b);
  const packet = f.programs.captureDraw(b, {...f.context, renderId: 2});
  assert.ok(!sampled(packet).includes(map)); assert.ok(values(packet).includes(.375));
  f.scene.environmentNode = N.texture(new THREE.Texture()); assert.equal(f.compile(), b);
  f.material.envNode = null; f.material.envMap = null;
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_ENVIRONMENT'});
});

test('non-node scene values retain the source ordinary-environment fallback', t => {
  const f = fixture(t);
  f.scene.environmentNode = {environment: 'unused'};
  const descriptor = f.compile(); assertSourceGraph(f, descriptor);
  assert.ok(sampled(f.programs.captureDraw(descriptor, f.context)).includes(f.scene.environment));
  f.scene.environmentNode = null;
  assert.equal(f.compile(), descriptor);
});

test('scene environments are inert for unlit, disabled and explicit fragment graphs', t => {
  const f = fixture(t);
  f.scene.environmentNode = new THREE.EnvironmentNode(N.vec3(1));
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_LIGHTS'});
  f.material.lights = false; assert.doesNotThrow(f.compile);
  f.material.lights = true; f.renderer.lighting.enabled = false; assert.doesNotThrow(f.compile);
  f.renderer.lighting.enabled = true; f.material.fragmentNode = N.vec4(.5, .25, .125, 1);
  assert.doesNotThrow(f.compile);
  const basic = new THREE.MeshBasicNodeMaterial();
  assert.doesNotThrow(() => f.programs.compile(basic, f.object, {...f.context, material: basic}));
});

test('scene material-reference and context-wrapped texture nodes preserve source sampling', t => {
  const f = fixture(t), atlas = cubeUV();
  f.material.map = atlas; f.scene.environmentNode = N.materialReference('map', 'texture');
  const a = f.compile(); assertSourceGraph(f, a);
  assert.ok(sampled(f.programs.captureDraw(a, f.context)).includes(atlas));
  f.scene.environmentNode = N.texture(atlas).context({getUV: () => N.vec2(.125, .25)});
  const b = f.compile(); assertSourceGraph(f, b);
  assert.doesNotMatch(b.fragment, /fn getFace/, 'a ContextNode remains authored sampling rather than raw PMREM conversion');
  assert.ok(sampled(f.programs.captureDraw(b, {...f.context, renderId: 2})).includes(atlas));
});
