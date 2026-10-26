/** Authored PBR environments execute the exact r186 graph and update lifecycle.
 * This is source construction/capture evidence, not browser pixel parity. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
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
  builder.lightsNode = N.lights([]); builder.environmentNode = N.texture(f.scene.environment);
  builder.build();
  assert.equal(descriptor.vertex, builder.vertexShader); assert.equal(descriptor.fragment, builder.fragmentShader);
}

for (const Material of [THREE.MeshStandardMaterial, THREE.MeshPhysicalMaterial,
  THREE.MeshStandardNodeMaterial, THREE.MeshPhysicalNodeMaterial]) {
  test(`${Material.name} procedural envNode overrides unused maps with exact source WGSL`, t => {
    const f = fixture(t, Material);
    f.material.envNode = N.vec3(.125, .25, .5);
    f.material.envMap = new THREE.Texture(); f.scene.environment = new THREE.Texture();
    f.scene.environmentNode = N.vec3(1, 0, 0);
    if (f.material.isMeshPhysicalMaterial || f.material.isMeshPhysicalNodeMaterial) {
      f.material.clearcoat = .5; f.material.sheen = .25; f.material.iridescence = .375;
    }
    const descriptor = f.compile(); assertSourceGraph(f, descriptor);
    const packet = f.programs.captureDraw(descriptor, f.context);
    assert.deepEqual(sampled(packet).map(texture => texture.name), ['DFG_LUT']);
    assert.match(descriptor.fragment, /iblIrradiance/); assert.doesNotMatch(descriptor.fragment, /fn getFace/);
    assert.ok(values(packet).includes(.375), 'the source material intensity accessor remains active');
  });
}

test('authored PMREM environments inherit radiance, roughness and material rotation contexts', t => {
  const f = fixture(t, THREE.MeshPhysicalNodeMaterial), atlas = cubeUV(0x3000);
  f.material.envNode = N.pmremTexture(atlas); f.material.clearcoat = .5; f.material.anisotropy = .25;
  f.material.envMapRotation.set(.2, .6, -.4);
  const descriptor = f.compile(); assertSourceGraph(f, descriptor);
  const packet = f.programs.captureDraw(descriptor, f.context), textures = sampled(packet);
  assert.ok(textures.includes(atlas)); assert.ok(!textures.includes(f.material.envMap));
  assert.ok(!textures.includes(f.scene.environment)); assert.equal(textures.length, 2);
  assert.match(descriptor.fragment, /roughnessToMip/); assert.match(descriptor.fragment, /clearcoatRadiance/);
  const rotation = new THREE.Matrix4().makeRotationFromEuler(f.material.envMapRotation).transpose().elements.map(Math.fround);
  const data = values(packet);
  assert.ok(data.some((_, i) => rotation.every((value, j) => data[i + j] === value)));
});

test('direct envNode textures rebuild source PMREM setup on compatible input replacement', t => {
  const f = fixture(t), initial = cubeUV(), input = N.texture(initial);
  f.material.envNode = input;
  let calls = 0;
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
  assert.equal(f.programs.diagnostics.programs, 1);
});

test('authored material references resolve their current source map before PMREM cache reuse', t => {
  const f = fixture(t), initial = f.material.envMap;
  f.material.envNode = N.materialReference('envMap', 'texture');
  const a = f.compile(), first = f.programs.captureDraw(a, f.context); assertSourceGraph(f, a);
  f.material.envMap = cubeUV(0x3000);
  assert.throws(() => f.programs.captureDraw(a, {...f.context, renderId: 2}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  const b = f.compile(), second = f.programs.captureDraw(b, {...f.context, renderId: 2});
  assert.notEqual(a, b); assert.ok(sampled(first).includes(initial)); assert.ok(sampled(second).includes(f.material.envMap));
  assertSourceGraph(f, b);
});

test('context-wrapped environment texture expressions retain their source sampling semantics', t => {
  const f = fixture(t), atlas = cubeUV();
  f.material.envNode = N.texture(atlas).context({getUV: () => N.vec2(.125, .25)});
  const descriptor = f.compile(); assertSourceGraph(f, descriptor);
  assert.doesNotMatch(descriptor.fragment, /fn getFace/, 'the source treats a ContextNode as an authored graph, not a raw PMREM input');
  assert.ok(sampled(f.programs.captureDraw(descriptor, f.context)).includes(atlas));
});

test('shared procedural environments update per logical object and preserve prior captures', t => {
  const f = fixture(t); let calls = 0;
  f.material.envNode = N.uniform(new THREE.Color()).onObjectUpdate(({object}) => { calls++; return object.userData.irradiance; });
  f.object.userData.irradiance = new THREE.Color(.1875, .3125, .4375);
  const other = new THREE.Mesh(f.object.geometry, f.material); other.userData.irradiance = new THREE.Color(.6875, .8125, .9375);
  f.scene.add(other); f.scene.updateMatrixWorld();
  const a = f.compile(), context = {...f.context, object: other};
  const b = f.programs.compile(f.material, other, context);
  const first = f.programs.captureDraw(a, f.context), saved = first.uniforms.map(bytes => bytes.slice());
  const second = f.programs.captureDraw(b, context);
  assert.equal(calls, 2); assert.ok(values(first).includes(.1875)); assert.ok(values(second).includes(.6875));
  assert.deepEqual(first.uniforms, saved); assert.equal(f.compile(), a);
  f.object.userData.irradiance.setRGB(.5625, .25, .125);
  const third = f.programs.captureDraw(a, {...f.context, renderId: 2});
  assert.equal(calls, 3); assert.ok(values(third).includes(.5625)); assert.deepEqual(first.uniforms, saved);
});

test('authored environment replacement and removal reject stale captures and restore map fallback', t => {
  const f = fixture(t); let calls = 0;
  f.material.colorNode = N.uniform(new THREE.Color()).onObjectUpdate(() => { calls++; return new THREE.Color(.5, .25, .125); });
  f.material.envNode = N.vec3(.125, .25, .5);
  const a = f.compile(); f.programs.captureDraw(a, f.context);
  f.material.envNode = N.vec3(.5, .25, .125);
  assert.throws(() => f.programs.captureDraw(a, {...f.context, renderId: 2}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(calls, 1);
  const b = f.compile(); f.programs.captureDraw(b, {...f.context, renderId: 2}); assert.notEqual(a, b);
  f.material.envNode = null;
  assert.throws(() => f.programs.captureDraw(b, {...f.context, renderId: 3}), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(calls, 2);
  const c = f.compile(); assert.notEqual(b, c);
  assert.ok(sampled(f.programs.captureDraw(c, {...f.context, renderId: 3})).includes(f.material.envMap));
});

test('authored envNode keeps material-versus-scene intensity changes as live uniforms', t => {
  const f = fixture(t), atlas = f.material.envMap;
  f.material.envNode = N.pmremTexture(atlas);
  const descriptor = f.compile(), first = f.programs.captureDraw(descriptor, f.context);
  f.material.envMap = null;
  assert.equal(f.compile(), descriptor);
  const second = f.programs.captureDraw(descriptor, {...f.context, renderId: 2});
  assert.ok(values(first).includes(.375)); assert.ok(values(second).includes(.0625));
  assert.deepEqual(sampled(first), sampled(second));
});

test('lighting-node injection and unsupported material families remain outside envNode admission', t => {
  const f = fixture(t);
  f.material.envNode = new THREE.EnvironmentNode(N.vec3(1));
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_LIGHTS'});
  const basic = new THREE.MeshBasicNodeMaterial(); basic.envNode = N.vec3(1);
  assert.throws(() => f.programs.compile(basic, f.object, {...f.context, material: basic}), {code: 'THREE_NODE_PROGRAM_GRAPH'});
});

test('ordinary material envNode selects native submission and preserves queued irradiance values', async t => {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureArrayLayers: 256});
  device.createTexture = descriptor => ({dimension: '2d', sampleCount: 1, ...descriptor,
    width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
    depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = descriptor => ({...descriptor}); device.queue.writeTexture = () => {};
  device.createRenderPipeline = descriptor => ({...descriptor, getBindGroupLayout: () => ({})});
  let configuration;
  const canvas = {width: 16, height: 16, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [16, 16], format: configuration.format, usage: 16}); }
  })};
  const Renderer = createWebGPURendererClass(THREE), renderer = new Renderer({canvas, device});
  t.after(() => renderer.dispose()); await renderer.init();
  let updates = 0, irradiance = new THREE.Color(.1875, .3125, .4375);
  const material = new THREE.MeshStandardMaterial({metalness: .25, roughness: .5});
  material.envNode = N.uniform(new THREE.Color()).setName('authoredEnvironmentIrradiance')
    .onObjectUpdate(() => { updates++; return irradiance; });
  const scene = new THREE.Scene(); scene.add(new THREE.Mesh(new THREE.PlaneGeometry(), material));
  const camera = new THREE.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  renderer.render(scene, camera);
  irradiance = new THREE.Color(.6875, .8125, .9375); renderer.render(scene, camera);
  assert.equal(updates, 2, 'ordinary material envNode runs once at each synchronous source draw');
  await renderer.waitForGPU();
  const draws = device.snapshots.flat().filter(draw => draw.pipeline.fragment?.module?.code?.includes('authoredEnvironmentIrradiance'));
  assert.equal(draws.length, 2);
  const uploaded = draw => [...draw.contents.values()].flatMap(bytes => [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);
  assert.ok(uploaded(draws[0]).includes(.1875)); assert.ok(!uploaded(draws[0]).includes(.6875));
  assert.ok(uploaded(draws[1]).includes(.6875)); assert.ok(!uploaded(draws[1]).includes(.1875));
  assert.equal(updates, 2, 'asynchronous preparation never repeats source graph updates');
  assert.equal(renderer._nodeCaptureBytes, 0);
});
