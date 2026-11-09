import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import SourceRenderObject from '../../upstream/three.js/src/renderers/common/RenderObject.js';
import SourceRenderObjects from '../../upstream/three.js/src/renderers/common/RenderObjects.js';
import SourceNodeManager from '../../upstream/three.js/src/renderers/common/nodes/NodeManager.js';
import {threeNodeBuilderMaterialKey} from './three_node_builder_cache.mjs';
import {createThreeNodePrograms} from './three_node_program.mjs';

function environment(value) {
  const texture = new T.DataTexture(new Uint16Array(336 * 64 * 4).fill(value), 336, 64, T.RGBAFormat, T.HalfFloatType);
  texture.mapping = T.CubeUVReflectionMapping; texture.minFilter = texture.magFilter = T.LinearFilter;
  texture.colorSpace = T.LinearSRGBColorSpace; texture.needsUpdate = true;
  return texture;
}
function fixture(t, Material = T.MeshStandardNodeMaterial) {
  const renderer = {library: new T.StandardNodeLibrary(), contextNode: T.TSL.context(),
    debug: {diagnostics: {keywords: false}, onNodeBuilderCreated: null}, depth: true, lighting: {enabled: true},
    shadowMap: {enabled: false, type: T.PCFShadowMap}, _currentSourceMaterial: null,
    backend: {isWebGPUBackend: true, hasFeature: () => false,
      utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})}},
    hasFeature: () => false, getRenderTarget: () => null, getOutputRenderTarget: () => null,
    getMRT: () => null, currentSamples: 1, _nodes: {nodeFrame: new T.NodeFrame()}, info: {calls: 1}};
  renderer.backend.createNodeBuilder = object => new T.WGSLNodeBuilder(object, renderer);
  const programs = createThreeNodePrograms({three: T, renderer});
  const nodes = new SourceNodeManager(renderer, renderer.backend);
  const source = new SourceRenderObjects(renderer, nodes, {}, {delete() {}}, {deleteForRender() {}}, {});
  t.after(() => { programs.dispose(); source.dispose(); });
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 30);
  camera.position.z = 4; camera.coordinateSystem = T.WebGPUCoordinateSystem;
  camera.updateProjectionMatrix(); camera.updateMatrixWorld();
  const lights = T.TSL.lights([]), renderContext = {id: 0};
  const add = (value, roughness = .5) => {
    const material = new Material({envMap: environment(value), roughness, metalness: .5});
    const object = new T.Mesh(new T.SphereGeometry(1, 8, 6), material); scene.add(object); scene.updateMatrixWorld();
    object.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, object.matrixWorld);
    object.normalMatrix.getNormalMatrix(object.modelViewMatrix);
    return {object, material};
  };
  const context = item => ({...item, scene, camera, lights: [], renderId: renderer.info.calls, frameId: 1});
  const compile = item => programs.compile(item.material, item.object, context(item));
  const capture = (item, descriptor = compile(item)) => programs.captureDraw(descriptor, context(item));
  const reference = item => {
    const object = source.get(item.object, item.material, scene, camera, lights, renderContext, null);
    return {object, state: object.getNodeBuilderState(), bindings: object.getBindings()};
  };
  return {renderer, programs, nodes, source, scene, camera, lights, renderContext, add, context, compile, capture, reference};
}
const sampled = packet => packet.textures.filter(binding => binding.isSampledTexture).map(binding => binding.texture);
const referenceTextures = reference => reference.bindings.flatMap(group => group.bindings)
  .filter(binding => binding.isSampledTexture).map(binding => binding.textureNode.value);
const values = packet => packet.uniforms.flatMap(bytes => [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);

for (const Material of [T.MeshStandardMaterial, T.MeshPhysicalMaterial, T.MeshStandardNodeMaterial, T.MeshPhysicalNodeMaterial]) {
  test(`${Material.name} source cache identity and shared PMREM nodes retain per-material uniforms`, t => {
    const f = fixture(t, Material), a = f.add(0x3800, .25), b = f.add(0x3400, .75);
    a.material.envMapIntensity = .375; b.material.envMapIntensity = .875;
    a.material.color.setRGB(.25, .5, .75); b.material.color.setRGB(.75, .5, .25);
    const sourceA = f.reference(a), sourceB = f.reference(b);
    for (const item of [a, b]) assert.equal(
      threeNodeBuilderMaterialKey(T, f.renderer, item.object, item.material, f.renderContext.id),
      SourceRenderObject.prototype.getMaterialCacheKey.call({object: item.object, geometry: item.object.geometry,
        material: item.material, renderer: f.renderer, context: f.renderContext, clippingContextCacheKey: '',
        getGeometryCacheKey: SourceRenderObject.prototype.getGeometryCacheKey}));
    assert.ok(sourceA.state === sourceB.state, 'the actual source NodeManager shares its builder state');
    assert.ok(referenceTextures(sourceB).includes(a.material.envMap));
    const descriptorA = f.compile(a), descriptorB = f.compile(b);
    assert.equal(descriptorA.vertex, sourceA.state.vertexShader); assert.equal(descriptorB.fragment, sourceB.state.fragmentShader);
    const first = f.capture(a, descriptorA), saved = first.uniforms.map(bytes => bytes.slice()), second = f.capture(b, descriptorB);
    assert.deepEqual(sampled(first).map(texture => texture.id), referenceTextures(sourceA).map(texture => texture.id));
    assert.deepEqual(sampled(second).map(texture => texture.id), referenceTextures(sourceB).map(texture => texture.id));
    assert.ok(values(first).includes(.375)); assert.ok(values(second).includes(.875));
    assert.ok(first.textures.find(binding => binding.isSampler)?.samplerIdentity !==
      second.textures.find(binding => binding.isSampler)?.samplerIdentity, 'binding wrappers are per source render object');
    b.material.envMapIntensity = .625; f.renderer.info.calls++;
    assert.equal(f.compile(b), descriptorB); assert.ok(values(f.capture(b, descriptorB)).includes(.625));
    assert.deepEqual(first.uniforms, saved, 'shared nodes do not rewrite accepted draw bytes');
  });
}

test('version-only invalidation and unmarked environment replacement retain the source builder', t => {
  const f = fixture(t), a = f.add(0x3800), b = f.add(0x3400), firstMap = a.material.envMap;
  const sourceA = f.reference(a), sourceB = f.reference(b), dA = f.compile(a), dB = f.compile(b);
  for (const mark of [false, true]) {
    b.material.envMap = environment(0x3000); if (mark) b.material.needsUpdate = true;
    f.renderer.info.calls++;
    assert.equal(f.reference(b).state, sourceB.state);
    assert.equal(f.compile(b), dB);
    assert.ok(sampled(f.capture(b, dB)).includes(firstMap));
  }
  b.material.roughness = 0; b.material.needsUpdate = true; f.renderer.info.calls++;
  const nextSource = f.reference(b), next = f.compile(b);
  assert.notEqual(nextSource.state, sourceA.state); assert.notEqual(next, dB);
  assert.deepEqual(sampled(f.capture(b, next)), referenceTextures(nextSource));
  assert.ok(sampled(f.capture(a, dA)).includes(firstMap));
});

test('shared source builders survive one material disposal and retire at their final user', t => {
  const f = fixture(t), a = f.add(0x3800), b = f.add(0x3400), firstMap = a.material.envMap;
  const firstSource = f.reference(a); f.reference(b); f.compile(a); const dB = f.compile(b);
  a.material.dispose(); f.renderer.info.calls++;
  const c = f.add(0x3000), sourceC = f.reference(c), dC = f.compile(c);
  assert.equal(sourceC.state, firstSource.state);
  assert.ok(sampled(f.capture(c, dC)).includes(firstMap));
  assert.ok(sampled(f.capture(b, dB)).includes(firstMap));
  b.material.dispose(); c.material.dispose(); f.renderer.info.calls++;
  const d = f.add(0x3a00), sourceD = f.reference(d), dD = f.compile(d);
  assert.notEqual(sourceD.state, firstSource.state);
  assert.ok(sampled(f.capture(d, dD)).includes(d.material.envMap));
  assert.equal(f.programs.diagnostics.programs, 1);
});

test('geometry disposal preserves builder state while object disposal releases its source reference', t => {
  const f = fixture(t), a = f.add(0x3800), firstMap = a.material.envMap;
  const source = f.reference(a), descriptor = f.compile(a);
  a.object.geometry.dispose(); a.material.envMap = environment(0x3000); a.material.needsUpdate = true; f.renderer.info.calls++;
  assert.equal(f.reference(a).state, source.state); assert.equal(f.compile(a), descriptor);
  assert.ok(sampled(f.capture(a, descriptor)).includes(firstMap));
  a.object.dispatchEvent({type: 'dispose'}); f.renderer.info.calls++;
  const nextSource = f.reference(a), next = f.compile(a);
  assert.notEqual(nextSource.state, source.state); assert.notEqual(next, descriptor);
  assert.ok(sampled(f.capture(a, next)).includes(a.material.envMap));
});

test('renderer context invalidation releases only its prior source graph generation', t => {
  const f = fixture(t), a = f.add(0x3800), b = f.add(0x3400);
  f.reference(a); f.reference(b); f.compile(a); const previous = f.compile(b);
  f.renderer.contextNode.needsUpdate = true; f.renderer.info.calls++;
  const source = f.reference(b), descriptor = f.compile(b);
  assert.notEqual(descriptor, previous);
  assert.deepEqual(sampled(f.capture(b, descriptor)), referenceTextures(source));
  assert.ok(sampled(f.capture(b, descriptor)).includes(b.material.envMap));
});

test('attachment contexts retain independent render objects and source builder lifetimes', t => {
  const f = fixture(t), a = f.add(0x3800), b = f.add(0x3400);
  const compile = (item, sourceRenderContext) => f.programs.compile(item.material, item.object,
    {...f.context(item), sourceRenderContext});
  const first = compile(a, 'default-default-0'), shared = compile(b, 'default-default-0');
  const other = compile(b, '1:1023:1016:0:true:false-default-0');
  assert.ok(sampled(f.capture(b, shared)).includes(a.material.envMap));
  assert.ok(sampled(f.capture(b, other)).includes(b.material.envMap));
  assert.equal(compile(a, 'default-default-0'), first);
  assert.equal(compile(b, 'default-default-0'), shared);
  assert.equal(f.programs.diagnostics.programs, 3);
});

test('leaving and reentering the shared profile retires its former source builder reference', t => {
  const f = fixture(t), a = f.add(0x3800), firstMap = a.material.envMap;
  const firstSource = f.reference(a), initial = f.compile(a);
  a.material.colorNode = T.TSL.vec3(.25, .5, .75); a.material.needsUpdate = true; f.renderer.info.calls++;
  const authoredSource = f.reference(a), authored = f.compile(a);
  assert.ok(authoredSource.state !== firstSource.state); assert.notEqual(authored.key, initial.key);
  assert.equal(f.programs.diagnostics.programs, 1);
  a.material.colorNode = null; a.material.envMap = environment(0x3000);
  a.material.needsUpdate = true; f.renderer.info.calls++;
  const nextSource = f.reference(a), next = f.compile(a);
  assert.ok(nextSource.state !== firstSource.state); assert.notEqual(next.key, authored.key);
  const textures = sampled(f.capture(a, next));
  assert.ok(!textures.includes(firstMap)); assert.ok(textures.includes(a.material.envMap));
  assert.deepEqual(textures.map(texture => texture.id), referenceTextures(nextSource).map(texture => texture.id));
  assert.equal(f.programs.diagnostics.programs, 1);
});
