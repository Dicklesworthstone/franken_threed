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

for (const Material of [T.MeshStandardMaterial, T.MeshPhysicalMaterial, T.MeshStandardNodeMaterial, T.MeshPhysicalNodeMaterial]) {
  test(`${Material.name} custom keys match source construction and invalidation calls`, t => {
    const f = fixture(t, Material), source = f.add(0x3800), candidate = f.add(0x3800);
    const key = function () { this.userData.calls++; return this.userData.variant; };
    for (const item of [source, candidate]) {
      item.material.userData = {calls: 0, variant: 'first'};
      item.material.customProgramCacheKey = key;
    }
    let reference, descriptor;
    for (const [change, calls, rebuilt] of [
      [() => {}, 1, true], [() => {}, 0, false],
      [material => { material.needsUpdate = true; }, 1, false],
      [material => { material.userData.variant = 'second'; material.needsUpdate = true; }, 2, true],
      [material => { material.userData.variant = 'unmarked'; }, 0, false],
    ]) {
      for (const item of [source, candidate]) { item.material.userData.calls = 0; change(item.material); }
      f.renderer.info.calls++;
      const nextReference = f.reference(source), nextDescriptor = f.compile(candidate);
      assert.equal(source.material.userData.calls, calls);
      assert.equal(candidate.material.userData.calls, calls);
      assert.equal(nextReference.object !== reference?.object, rebuilt);
      assert.equal(nextDescriptor !== descriptor, rebuilt);
      reference = nextReference; descriptor = nextDescriptor;
    }
  });
}

test('custom key comparison errors retain the old source owner and reconstruction errors retire it', t => {
  const f = fixture(t), source = f.add(0x3800), candidate = f.add(0x3800), error = new Error('key failure');
  const key = function () {
    if (++this.userData.calls === this.userData.throwAt) throw error;
    return this.userData.variant;
  };
  for (const item of [source, candidate]) {
    item.material.userData = {calls: 0, variant: 'first', throwAt: -1};
    item.material.customProgramCacheKey = key;
  }
  const initial = f.reference(source); f.compile(candidate);
  for (const [throwAt, alive] of [[1, true], [2, false]]) {
    for (const item of [source, candidate]) {
      Object.assign(item.material.userData, {calls: 0, variant: 'replacement', throwAt});
      item.material.needsUpdate = true;
    }
    f.renderer.info.calls++;
    assert.throws(() => f.reference(source), value => value === error);
    assert.throws(() => f.compile(candidate), value => value === error);
    assert.equal(source.material.userData.calls, throwAt); assert.equal(candidate.material.userData.calls, throwAt);
    assert.equal(f.source._renderObjects.has(initial.object), alive);
    assert.equal(f.programs.diagnostics.programs, alive ? 1 : 0);
  }
  for (const item of [source, candidate]) Object.assign(item.material.userData, {calls: 0, throwAt: -1});
  f.renderer.info.calls++;
  f.reference(source); f.compile(candidate);
  assert.equal(source.material.userData.calls, 1); assert.equal(candidate.material.userData.calls, 1);
});

test('custom keys choose shared environment builders and isolate newly invalidated variants', t => {
  const f = fixture(t), a = f.add(0x3800), b = f.add(0x3400);
  const key = function () { return this.userData.variant; };
  for (const item of [a, b]) { item.material.customProgramCacheKey = key; item.material.userData.variant = 'shared'; }
  const sourceA = f.reference(a), sourceB = f.reference(b), first = f.compile(a), second = f.compile(b);
  assert.equal(sourceA.state, sourceB.state);
  assert.deepEqual(sampled(f.capture(b, second)), referenceTextures(sourceB));
  b.material.userData.variant = 'separate'; b.material.needsUpdate = true; f.renderer.info.calls++;
  const sourceNext = f.reference(b), next = f.compile(b);
  assert.notEqual(sourceNext.state, sourceA.state); assert.notEqual(next, second);
  assert.deepEqual(sampled(f.capture(b, next)), referenceTextures(sourceNext));
  assert.ok(sampled(f.capture(a, first)).includes(a.material.envMap));
  assert.ok(sampled(f.capture(b, next)).includes(b.material.envMap));
});

test('custom key callback mutations preserve constructor version and selected environment', t => {
  const f = fixture(t), source = f.add(0x3800), candidate = f.add(0x3800), selected = environment(0x3400);
  const key = function () {
    if (++this.userData.calls === 1) { this.needsUpdate = true; this.envMap = this.userData.selected; }
    return 'selected';
  };
  for (const item of [source, candidate]) {
    item.material.userData = {calls: 0, selected}; item.material.customProgramCacheKey = key;
  }
  const first = f.reference(source), descriptor = f.compile(candidate);
  assert.deepEqual(sampled(f.capture(candidate, descriptor)), referenceTextures(first));
  assert.ok(sampled(f.capture(candidate, descriptor)).includes(selected));
  for (const expected of [2, 2]) {
    f.renderer.info.calls++;
    assert.equal(f.reference(source).object, first.object); assert.equal(f.compile(candidate), descriptor);
    assert.equal(source.material.userData.calls, expected); assert.equal(candidate.material.userData.calls, expected);
  }
});

test('installing a custom key without invalidation retains the accepted source cache', t => {
  const f = fixture(t), source = f.add(0x3800), candidate = f.add(0x3800);
  const initial = f.reference(source), descriptor = f.compile(candidate);
  const key = function () { this.userData.calls++; return 'changed'; };
  for (const item of [source, candidate]) { item.material.userData.calls = 0; item.material.customProgramCacheKey = key; }
  f.renderer.info.calls++;
  assert.equal(f.reference(source).object, initial.object); assert.equal(f.compile(candidate), descriptor);
  assert.equal(source.material.userData.calls, 0); assert.equal(candidate.material.userData.calls, 0);
  for (const item of [source, candidate]) item.material.needsUpdate = true;
  f.renderer.info.calls++;
  assert.notEqual(f.reference(source).object, initial.object); assert.notEqual(f.compile(candidate), descriptor);
  assert.equal(source.material.userData.calls, 2); assert.equal(candidate.material.userData.calls, 2);
});

test('custom key effects cannot publish a graph outside the admitted material profile', t => {
  const f = fixture(t), item = f.add(0x3800); let calls = 0;
  item.material.customProgramCacheKey = function () { calls++; this.colorNode = T.TSL.vec3(1, 0, 0); return 'authored'; };
  assert.throws(() => f.compile(item), {code: 'THREE_NODE_PROGRAM_HOOK'});
  assert.equal(calls, 1); assert.ok(item.material.colorNode, 'callback effects remain visible after rejection');
  assert.equal(f.programs.diagnostics.programs, 0);
});

for (const field of ['context', 'shadows']) test(`custom key ${field} mutations use the source dynamic-cache boundary`, t => {
  const sourceFixture = fixture(t), candidateFixture = fixture(t);
  const source = sourceFixture.add(0x3800), candidate = candidateFixture.add(0x3800);
  const key = function () {
    if (++this.userData.calls === 1) {
      if (this.userData.field === 'context') this.userData.renderer.contextNode.needsUpdate = true;
      else this.userData.renderer.shadowMap.enabled = true;
    }
    return 'dynamic';
  };
  for (const [f, item] of [[sourceFixture, source], [candidateFixture, candidate]]) {
    item.material.userData = {calls: 0, field, renderer: f.renderer}; item.material.customProgramCacheKey = key;
  }
  let previousSource, previousCandidate;
  for (let turn = 0; turn < 3; turn++) {
    sourceFixture.renderer.info.calls++; candidateFixture.renderer.info.calls++;
    const reference = sourceFixture.reference(source), descriptor = candidateFixture.compile(candidate);
    assert.equal(candidate.material.userData.calls, source.material.userData.calls);
    assert.equal(descriptor !== previousCandidate, reference.object !== previousSource);
    previousSource = reference.object; previousCandidate = descriptor;
  }
});

test('source key callbacks precede material-key getters and return-value coercion', t => {
  const f = fixture(t), source = f.add(0x3800), candidate = f.add(0x3800);
  const key = function () {
    this.userData.events.push('key'); this.addedByKey = 1;
    return { [Symbol.toPrimitive]: hint => { this.userData.events.push(`coerce-${hint}`); return 'value'; } };
  };
  for (const item of [source, candidate]) {
    const events = []; item.material.userData.events = events; item.material.customProgramCacheKey = key;
    Object.defineProperty(item.material, 'constructor', {get() { events.push('constructor'); return T.MeshStandardNodeMaterial; }});
  }
  const reference = SourceRenderObject.prototype.getMaterialCacheKey.call({object: source.object, geometry: source.object.geometry,
    material: source.material, renderer: f.renderer, context: f.renderContext, clippingContextCacheKey: '',
    getGeometryCacheKey: SourceRenderObject.prototype.getGeometryCacheKey});
  const actual = threeNodeBuilderMaterialKey(T, f.renderer, candidate.object, candidate.material, f.renderContext.id);
  assert.equal(actual, reference); assert.deepEqual(candidate.material.userData.events, source.material.userData.events);
  assert.equal(candidate.material.userData.events[0], 'key');
});

test('an override custom-key owner retires when its original source material is disposed', t => {
  const f = fixture(t), item = f.add(0x3800), original = new T.MeshStandardMaterial(); let calls = 0;
  item.material.customProgramCacheKey = () => { calls++; return 'override'; };
  f.renderer._currentSourceMaterial = original;
  const source = f.reference(item), descriptor = f.compile(item);
  original.dispose();
  assert.equal(f.source._renderObjects.has(source.object), false);
  assert.equal(f.programs.diagnostics.programs, 0);
  calls = 0; f.renderer.info.calls++;
  assert.notEqual(f.reference(item).object, source.object); assert.notEqual(f.compile(item), descriptor);
  assert.equal(calls, 2, 'each replacement source/candidate render object constructs its key once');
});

for (const owner of ['material', 'object', 'override']) test(`disposing ${owner} during key comparison keeps only the current source draw`, t => {
  const f = fixture(t), source = f.add(0x3800), candidate = f.add(0x3800), selected = environment(0x3400);
  const originals = [new T.MeshStandardMaterial(), new T.MeshStandardMaterial()];
  candidate.material.envMap = source.material.envMap;
  const key = function () {
    this.userData.calls++;
    if (this.userData.disposeNow) {
      this.userData.disposeNow = false; this.envMap = selected;
      this.userData.owner.dispatchEvent({type: 'dispose'});
    }
    return 'unchanged';
  };
  for (const [index, item] of [source, candidate].entries()) {
    item.material.userData = {calls: 0, disposeNow: false,
      owner: owner === 'material' ? item.material : owner === 'object' ? item.object : originals[index]};
    item.material.customProgramCacheKey = key;
  }
  const reference = () => {
    f.renderer._currentSourceMaterial = owner === 'override' ? originals[0] : null;
    return f.reference(source);
  };
  const compile = () => {
    f.renderer._currentSourceMaterial = owner === 'override' ? originals[1] : null;
    return f.compile(candidate);
  };
  const initialSource = reference(), initial = compile(), firstEnvironment = source.material.envMap;
  for (const item of [source, candidate]) { item.material.userData.disposeNow = true; item.material.needsUpdate = true; }
  f.renderer.info.calls++;
  const sourceDraw = reference(), draw = compile();
  assert.equal(sourceDraw.object, initialSource.object); assert.equal(draw, initial);
  assert.equal(f.source._renderObjects.size, 0); assert.equal(f.programs.diagnostics.programs, 0);
  assert.deepEqual([source.material.userData.calls, candidate.material.userData.calls], [2, 2]);
  assert.deepEqual(sampled(f.capture(candidate, draw)), referenceTextures(sourceDraw));
  assert.ok(referenceTextures(sourceDraw).includes(firstEnvironment));
  f.renderer.info.calls++;
  const nextSource = reference(), next = compile();
  assert.notEqual(nextSource.object, initialSource.object); assert.notEqual(next, initial);
  assert.deepEqual([source.material.userData.calls, candidate.material.userData.calls], [3, 3]);
  assert.deepEqual(sampled(f.capture(candidate, next)), referenceTextures(nextSource));
  assert.ok(sampled(f.capture(candidate, next)).includes(selected));
});
