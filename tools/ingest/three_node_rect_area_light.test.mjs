/** Exact r186 area-light shaders, LTC selection and camera-dependent updates.
 * The upstream texture library is retained source data, not a renderer. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {RectAreaLightTexturesLib} from './fixtures/three_rect_area_light.mjs';
const N = THREE.TSL;
const ltc = RectAreaLightTexturesLib.init(); THREE.RectAreaLightNode.setLTC(ltc);
// The pinned builder caches feature availability at module scope. A second
// source instance gives the float-capable device the independent first feature
// query it would receive in another application/realm, without patching source.
const floatSource = await import('../../upstream/three.js/build/three.webgpu.js?rect-area-float-device');
floatSource.RectAreaLightNode.setLTC(ltc);

function fixture(t, Material = THREE.MeshStandardNodeMaterial, filterable = false) {
  const T = filterable ? floatSource : THREE, nodes = T.TSL;
  const available = name => name === 'float32-filterable' && filterable;
  const renderer = {library: new T.StandardNodeLibrary(), contextNode: nodes.context(),
    debug: {diagnostics: {keywords: false}}, depth: true, lighting: {enabled: true},
    backend: {hasFeature: available, utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})}},
    hasFeature: available, getRenderTarget: () => null, getMRT: () => null, currentSamples: 1,
    _nodes: {nodeFrame: new T.NodeFrame()}, info: {calls: 1}};
  const programs = createThreeNodePrograms({three: T, renderer}); t.after(() => programs.dispose());
  const material = new T[Material.name]();
  if ('metalness' in material) { material.metalness = .25; material.roughness = .5; }
  const object = new T.Mesh(new T.SphereGeometry(1, 8, 6), material), scene = new T.Scene();
  const light = new T.RectAreaLight(new T.Color(.25, .5, .75), 3.5, 4, 6);
  light.position.set(1.25, 2.5, 3.75); light.rotation.set(.2, .4, .1);
  scene.add(object, light);
  const camera = new T.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 5;
  camera.coordinateSystem = T.WebGPUCoordinateSystem; camera.updateProjectionMatrix();
  scene.updateMatrixWorld(); camera.updateMatrixWorld();
  const context = {scene, camera, lights: [light], object, material, renderId: 1, frameId: 1};
  return {three: T, renderer, programs, material, object, scene, camera, light, context,
    compile: () => programs.compile(material, object, context)};
}
const sampled = packet => packet.textures.filter(binding => binding.isSampledTexture).map(binding => binding.texture);
const values = packet => packet.uniforms.flatMap(bytes => [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);
const contains = (packet, vector) => values(packet).some((_, index, all) => vector.every((v, j) => all[index + j] === Math.fround(v)));
function sourceGraph(f, descriptor) {
  const T = f.three, builder = new T.WGSLNodeBuilder(f.object, f.renderer);
  builder.material = f.material; builder.context.material = f.material;
  builder.scene = f.scene; builder.camera = f.context.camera; builder.lightsNode = T.TSL.lights(f.context.lights); builder.build();
  // Source Fn caching can reorder whole helper declarations on a later build.
  // Compare every byte of their bodies and of the main shader independently of
  // that declaration order; no constants, expressions or variable names change.
  const canonical = shader => shader.replace(/(\/\/ codes\n)([\s\S]*?)(?=@fragment)/, (_, prefix, functions) =>
    prefix + functions.split(/(?=^fn )/m).map(part => part.trim()).filter(Boolean).sort().join('\n\n') + '\n\n');
  assert.equal(descriptor.vertex, builder.vertexShader); assert.equal(canonical(descriptor.fragment), canonical(builder.fragmentShader));
  return [...builder.nodes].find(node => node instanceof T.RectAreaLightNode);
}
function sourceLightState(light, camera) {
  const node = new THREE.RectAreaLightNode(light);
  node.update({camera});
  return [node.halfWidth.value.toArray(), node.halfHeight.value.toArray(), node.color.toArray()];
}

for (const Material of [THREE.MeshStandardMaterial, THREE.MeshPhysicalMaterial,
  THREE.MeshStandardNodeMaterial, THREE.MeshPhysicalNodeMaterial]) {
  for (const filterable of [false, true]) {
    test(`${Material.name} retains exact area-light WGSL and ${filterable ? 'float' : 'half-float'} LTC tables`, t => {
      const f = fixture(t, Material, filterable);
      if (f.material.isMeshPhysicalMaterial || f.material.isMeshPhysicalNodeMaterial) {
        f.material.clearcoat = .5; f.material.clearcoatRoughness = .25;
      }
      const descriptor = f.compile(); sourceGraph(f, descriptor);
      const packet = f.programs.captureDraw(descriptor, f.context), textures = sampled(packet);
      assert.match(descriptor.fragment, /fn LTC_Evaluate/); assert.match(descriptor.fragment, /directSpecular/);
      const expected = filterable ? [ltc.LTC_FLOAT_1, ltc.LTC_FLOAT_2] : [ltc.LTC_HALF_1, ltc.LTC_HALF_2];
      for (const texture of expected) {
        assert.ok(textures.includes(texture)); assert.equal(texture.image.width, 64); assert.equal(texture.image.height, 64);
        assert.equal(texture.image.data.length, 64 * 64 * 4);
      }
      const rejected = filterable ? [ltc.LTC_HALF_1, ltc.LTC_HALF_2] : [ltc.LTC_FLOAT_1, ltc.LTC_FLOAT_2];
      assert.ok(rejected.every(texture => !textures.includes(texture)));
      for (const vector of sourceLightState(f.light, f.camera)) assert.ok(contains(packet, vector));
    });
  }
}

test('area-light extent, orientation and radiance stay live without rewriting earlier packets', t => {
  const f = fixture(t), descriptor = f.compile(), first = f.programs.captureDraw(descriptor, f.context);
  const saved = first.uniforms.map(bytes => bytes.slice());
  f.light.width = 7.5; f.light.height = 2.25; f.light.rotation.set(.6, -.3, .4);
  f.light.position.set(-1.5, 3.25, 2.75); f.light.intensity = .625; f.light.color.setRGB(.5, .25, .875);
  f.scene.updateMatrixWorld(); assert.equal(f.compile(), descriptor);
  const second = f.programs.captureDraw(descriptor, {...f.context, renderId: 2});
  for (const vector of sourceLightState(f.light, f.camera)) assert.ok(contains(second, vector));
  const position = f.light.position.clone().applyMatrix4(f.camera.matrixWorldInverse);
  assert.ok(contains(second, position.toArray()));
  assert.notDeepEqual(first.uniforms, second.uniforms); assert.deepEqual(first.uniforms, saved);
});

test('shared area lights keep source RENDER frequency across objects, rebuilds and camera changes', t => {
  const f = fixture(t), descriptor = f.compile(), reference = sourceGraph(f, descriptor);
  assert.equal(reference.updateType, THREE.NodeUpdateType.RENDER);
  f.programs.captureDraw(descriptor, f.context);
  const initialWidth = reference.halfWidth.value.clone();
  f.light.width = 8;
  const other = new THREE.Mesh(f.object.geometry, f.material); f.scene.add(other); f.scene.updateMatrixWorld();
  const otherDescriptor = f.programs.compile(f.material, other, {...f.context, object: other});
  f.programs.captureDraw(otherDescriptor, {...f.context, object: other});
  assert.deepEqual(reference.halfWidth.value, initialWidth, 'another object in the same render does not update shared light uniforms');
  f.material.needsUpdate = true; const rebuilt = f.compile(); f.programs.captureDraw(rebuilt, f.context);
  assert.deepEqual(reference.halfWidth.value, initialWidth, 'rebuilding the program retains the source light node and marker');
  const camera = new THREE.PerspectiveCamera(50, 1, .1, 30); camera.position.set(3, 1, 5); camera.lookAt(0, 0, 0);
  camera.coordinateSystem = THREE.WebGPUCoordinateSystem; camera.updateProjectionMatrix(); camera.updateMatrixWorld();
  const next = f.programs.captureDraw(rebuilt, {...f.context, camera, renderId: 2});
  for (const vector of sourceLightState(f.light, camera)) assert.ok(contains(next, vector));
  assert.notDeepEqual(reference.halfWidth.value, initialWidth, 'a new render/camera updates even within the same animation frame');
});

test('multiple area lights share source LTC bindings while retaining distinct light uniforms', t => {
  const f = fixture(t), second = new THREE.RectAreaLight(new THREE.Color(.125, .375, .625), 1.25, 2.5, 3.5);
  second.position.set(-3, 2, 4); second.rotation.y = -.25; f.scene.add(second); f.scene.updateMatrixWorld();
  f.context.lights.push(second);
  const descriptor = f.compile(); sourceGraph(f, descriptor);
  const packet = f.programs.captureDraw(descriptor, f.context), textures = sampled(packet);
  assert.equal(textures.filter(texture => texture === ltc.LTC_HALF_1).length, 1);
  assert.equal(textures.filter(texture => texture === ltc.LTC_HALF_2).length, 1);
  for (const light of [f.light, second]) for (const vector of sourceLightState(light, f.camera)) assert.ok(contains(packet, vector));
});

test('non-PBR source models do not invent area-light contributions', t => {
  for (const Material of [THREE.MeshBasicNodeMaterial, THREE.MeshLambertNodeMaterial, THREE.MeshPhongNodeMaterial, THREE.MeshToonNodeMaterial]) {
    const f = fixture(t, Material); const descriptor = f.compile();
    if (Material !== THREE.MeshBasicNodeMaterial) sourceGraph(f, descriptor);
    assert.doesNotMatch(descriptor.fragment, /LTC_Evaluate/);
    assert.ok(sampled(f.programs.captureDraw(descriptor, f.context)).every(texture =>
      texture !== ltc.LTC_HALF_1 && texture !== ltc.LTC_HALF_2));
  }
});

test('area lights preserve the source LTC initialization prerequisite and exact library registration', t => {
  const f = fixture(t); THREE.RectAreaLightNode.setLTC(null);
  try {
    for (const Material of [THREE.MeshStandardNodeMaterial, THREE.MeshLambertNodeMaterial]) {
      const source = fixture(t, Material), builder = new THREE.WGSLNodeBuilder(source.object, source.renderer);
      builder.scene = source.scene; builder.camera = source.camera; builder.lightsNode = N.lights([source.light]);
      assert.throws(() => builder.build(), TypeError);
      assert.throws(source.compile, {code: 'THREE_NODE_PROGRAM_BUILD'});
    }
  }
  finally { THREE.RectAreaLightNode.setLTC(ltc); }
  const descriptor = f.compile(); assert.match(descriptor.fragment, /LTC_Evaluate/);
  const original = f.renderer.library.getLightNodeClass.bind(f.renderer.library);
  class Replacement extends THREE.RectAreaLightNode {}
  f.renderer.library.getLightNodeClass = Source => Source === THREE.RectAreaLight ? Replacement : original(Source);
  assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_LIGHTS'});
});
