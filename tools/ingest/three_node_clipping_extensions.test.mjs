import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/src/Three.WebGPU.js';
import SourceClippingContext from '../../upstream/three.js/src/renderers/common/ClippingContext.js';
import SourceRenderObjects from '../../upstream/three.js/src/renderers/common/RenderObjects.js';
import SourceNodeManager from '../../upstream/three.js/src/renderers/common/nodes/NodeManager.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {captureThreeNodeClipping} from './three_node_clipping.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL;
const canonical = shader => shader.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
const unpack = (descriptor, packet) => descriptor.groups.flatMap((group, index) =>
  group.uniforms.map(member => packet.uniforms[index].slice(member.offset, member.offset + member.byteLength)));
function fixture(t, family = 'MeshStandard', registered = false) {
  const trace = [];
  class Extension extends T[family + 'NodeMaterial'] {
    constructor() { super(); this.applicationGain = .375; this.applicationVariant = 0; }
    customProgramCacheKey() { trace.push('key'); return super.customProgramCacheKey() + ':' + this.applicationVariant; }
    setupOutput(builder, output) {
      trace.push(['setup', builder.object]);
      return super.setupOutput(builder, N.vec4(output.rgb.mul(N.materialReference('applicationGain', 'float')), output.a));
    }
  }
  class Application extends T[family + 'Material'] {
    constructor() { super(); this.type = 'ClippedApplication'; this.applicationGain = .375; this.applicationVariant = 0; }
    customProgramCacheKey() { trace.push('key'); return super.customProgramCacheKey() + ':' + this.applicationVariant; }
  }
  const renderer = {library: new T.StandardNodeLibrary(), contextNode: N.context(),
    debug: {diagnostics: {keywords: false}, onNodeBuilderCreated: null}, depth: true, lighting: {enabled: true},
    shadowMap: {enabled: false, type: T.PCFShadowMap}, _currentSourceMaterial: null,
    backend: {isWebGPUBackend: true, hasFeature: () => false,
      utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})}},
    hasFeature: () => false, getRenderTarget: () => null, getOutputRenderTarget: () => null,
    getMRT: () => null, currentSamples: 1, _nodes: {nodeFrame: new T.NodeFrame()}, info: {calls: 1}};
  renderer.backend.createNodeBuilder = object => new T.WGSLNodeBuilder(object, renderer);
  if (registered) renderer.library.addMaterial(Extension, 'ClippedApplication');
  const programs = createThreeNodePrograms({three: T, renderer});
  const nodes = new SourceNodeManager(renderer, renderer.backend);
  const objects = new SourceRenderObjects(renderer, nodes, {}, {delete() {}}, {deleteForRender() {}}, {});
  t.after(() => { programs.dispose(); objects.dispose(); });
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 30);
  camera.position.z = 4; camera.coordinateSystem = T.WebGPUCoordinateSystem;
  camera.updateProjectionMatrix(); camera.updateMatrixWorld();
  const lights = N.lights([]), renderContext = {id: 0}, clipping = new SourceClippingContext();
  clipping.updateGlobal(scene, camera);
  const material = new (registered ? Application : Extension)(), geometry = new T.PlaneGeometry();
  const group = (normal = new T.Vector3(1, 0, 0), constant = .25, parent = scene) => {
    const value = new T.ClippingGroup(); value.clippingPlanes = [new T.Plane(normal, constant)]; parent.add(value); return value;
  };
  const add = (parent, value = material) => {
    const object = new T.Mesh(geometry, value); parent.add(object); scene.updateMatrixWorld();
    object.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, object.matrixWorld);
    object.normalMatrix.getNormalMatrix(object.modelViewMatrix); return {object, material: value};
  };
  const context = (item, projected) => ({...item, scene, camera, lights: lights.getLights(), sourceLightsNode: lights,
    renderId: renderer.info.calls, frameId: renderer.info.calls, clippingContext: projected});
  const compare = (item, projected) => {
    const source = objects.get(item.object, item.material, scene, camera, lights, renderContext, projected);
    const reference = source.getNodeBuilderState(), expected = trace.splice(0);
    const descriptor = programs.compile(item.material, item.object, context(item, projected));
    assert.deepEqual(trace.splice(0), expected, 'cache keys and authored setup retain source call counts and order');
    assert.equal(canonical(descriptor.vertex), canonical(reference.vertexShader));
    assert.equal(canonical(descriptor.fragment), canonical(reference.fragmentShader));
    return {descriptor, reference, source};
  };
  const capture = (item, projected, descriptor) => programs.captureDraw(descriptor, context(item, projected));
  const referenceDraw = source => {
    nodes.nodeFrame.frameId = nodes.nodeFrame.renderId = renderer.info.calls;
    nodes.updateBefore(source); nodes.updateForRender(source);
    const uniforms = source.getBindings().flatMap(group => group.bindings).filter(binding => binding.isUniformBuffer).map(binding => {
      binding.update();
      const bytes = new Uint8Array(binding.byteLength), array = binding.buffer;
      bytes.set(new Uint8Array(array.buffer, array.byteOffset, array.byteLength)); return bytes;
    });
    nodes.updateAfter(source); return uniforms;
  };
  return {renderer, programs, nodes, objects, scene, camera, material, geometry, clipping, trace,
    lights, renderContext, group, add, context, compare, capture, referenceDraw};
}

for (const family of ['MeshBasic', 'MeshToon', 'MeshLambert', 'MeshPhong', 'MeshStandard', 'MeshPhysical'])
for (const registered of [false, true])
test(`${registered ? 'registered' : 'inherited'} ${family} extensions retain per-group clipping construction`, t => {
  const f = fixture(t, family, registered), left = f.group(), right = f.group(new T.Vector3(0, 1, 0), .625);
  const a = f.add(left), b = f.add(left), c = f.add(right);
  const leftContext = f.clipping.getGroupContext(left), rightContext = f.clipping.getGroupContext(right);
  const first = f.compare(a, leftContext), shared = f.compare(b, leftContext), separate = f.compare(c, rightContext);
  assert.equal(first.reference, shared.reference, 'one source clipping group shares its builder');
  assert.notEqual(first.reference, separate.reference, 'different groups retain distinct captured plane arrays');
  assert.equal(f.programs.diagnostics.programs, 3, 'each source render object keeps its own binding packets');
  const saved = f.capture(a, leftContext, first.descriptor);
  assert.deepEqual(unpack(first.descriptor, saved), f.referenceDraw(first.source), 'captured plane and material uploads match the source');
  const bytes = saved.uniforms.map(value => value.slice());
  left.clippingPlanes[0].constant = .875; f.renderer.info.calls++;
  f.clipping.getGroupContext(left);
  assert.equal(f.compare(a, leftContext).descriptor, first.descriptor, 'moving planes do not reconstruct a shader');
  const updated = f.capture(a, leftContext, first.descriptor);
  assert.deepEqual(unpack(first.descriptor, updated), f.referenceDraw(first.source));
  assert.deepEqual(saved.uniforms, bytes, 'accepted draw packets retain their old plane coefficients');
  assert.notDeepEqual(updated.uniforms, saved.uniforms);
  right.add(a.object); f.renderer.info.calls++;
  const moved = f.compare(a, rightContext);
  assert.notEqual(moved.descriptor, first.descriptor, 'moving between equal-sized clipping groups invalidates the render object');
  assert.equal(moved.reference, separate.reference, 'the destination group keeps its existing source builder');
});

for (const registered of [false, true]) for (const samples of [1, 4])
test(`clipped extensions retain nested union/intersection invalidation and coverage (${registered}, ${samples}x)`, t => {
  const f = fixture(t, 'MeshStandard', registered), outer = f.group(), inner = f.group(new T.Vector3(0, 1, 0), .5, outer);
  f.renderer.currentSamples = samples; f.material.alphaToCoverage = true;
  inner.clipIntersection = true; const item = f.add(inner);
  const project = () => f.clipping.getGroupContext(outer).getGroupContext(inner);
  let projected = project(), previous = f.compare(item, projected);
  const capture = () => {
    const packet = f.capture(item, projected, previous.descriptor);
    assert.deepEqual(unpack(previous.descriptor, packet), f.referenceDraw(previous.source)); return packet;
  };
  const original = capture(), saved = original.uniforms.map(bytes => bytes.slice());
  for (const change of [
    () => { outer.clippingPlanes.push(new T.Plane(new T.Vector3(0, 0, 1), .75)); },
    () => { inner.clipIntersection = false; },
    () => { outer.clippingPlanes = []; },
    () => { inner.clippingPlanes = []; },
    () => { inner.clippingPlanes = [new T.Plane(new T.Vector3(1, 0, 0), .125)]; },
  ]) {
    change(); f.renderer.info.calls++; projected = project();
    assert.throws(() => f.capture(item, projected, previous.descriptor), {code: 'THREE_NODE_PROGRAM_CHANGED'});
    const next = f.compare(item, projected);
    assert.notEqual(next.descriptor, previous.descriptor, 'source layout changes construct fresh uniforms and shaders');
    previous = next; capture();
  }
  assert.deepEqual(original.uniforms, saved);
  f.material.applicationGain = .875; f.material.needsUpdate = true;
  assert.equal(f.compare(item, projected).descriptor, previous.descriptor, 'unchanged source key retains the clipped builder');
  f.material.applicationVariant++; f.material.needsUpdate = true;
  const replacement = f.compare(item, projected);
  assert.notEqual(replacement.descriptor, previous.descriptor);
  f.material.dispose(); assert.equal(f.programs.diagnostics.programs, 0);
  f.compare(item, projected);
});

test('clipping projection retains source group keys per render context and freezes accepted inherited planes', t => {
  const f = fixture(t), outer = f.group(), inner = f.group(new T.Vector3(0, 1, 0), .5, outer), item = f.add(inner);
  inner.clipIntersection = true;
  const project = (renderContext = 'canvas', contextOwner = f.renderer) => {
    const hierarchy = new Map(), visibility = new Map();
    f.scene.traverse(object => { hierarchy.set(object, object.children.slice()); visibility.set(object, object.visible); });
    return captureThreeNodeClipping(T, {root: f.scene, camera: f.camera, hierarchy, visibility, contextOwner, renderContext,
      fail: (_, message) => assert.fail(message)}).get(item.object);
  };
  const source = () => f.clipping.getGroupContext(outer).getGroupContext(inner);
  const planes = context => [context.intersectionPlanes, context.unionPlanes].map(list => list.map(plane => plane.toArray()));
  const normalize = key => key === '' ? key : key.slice(key.indexOf(':'));
  let expected = source(), actual = project();
  assert.deepEqual(planes(actual), planes(expected)); assert.equal(normalize(actual.cacheKey), normalize(expected.cacheKey));
  const original = actual, saved = planes(actual), initialKey = actual.cacheKey;
  outer.clippingPlanes[0].constant = .875; f.camera.position.x = .5; f.camera.updateMatrixWorld();
  f.clipping.updateGlobal(f.scene, f.camera); expected = source(); actual = project();
  assert.equal(actual.cacheKey, initialKey, 'coefficient and camera changes keep the context identity');
  assert.deepEqual(planes(actual), planes(expected)); assert.deepEqual(planes(original), saved);
  assert.notEqual(project('target').cacheKey, initialKey, 'different source render contexts own different plane arrays');
  assert.notEqual(project('canvas', {}).cacheKey, initialKey, 'renderer ownership never shares clipping state');
  for (const change of [
    () => { outer.clippingPlanes.push(new T.Plane(new T.Vector3(0, 0, 1), .75)); },
    () => { inner.clipIntersection = false; },
    () => { inner.clippingPlanes = []; },
    () => { outer.clippingPlanes = []; },
    () => { inner.clipIntersection = true; inner.clippingPlanes = [new T.Plane(new T.Vector3(1, 0, 0), .25)]; },
  ]) {
    const oldActual = actual.cacheKey, oldExpected = expected.cacheKey;
    change(); expected = source(); actual = project();
    assert.equal(actual.cacheKey === oldActual, expected.cacheKey === oldExpected, 'source key transitions retain their exact invalidation boundaries');
    assert.equal(normalize(actual.cacheKey), normalize(expected.cacheKey));
    assert.deepEqual(planes(actual), planes(expected));
  }
});

test('disposing program ownership recreates clipping contexts without changing accepted snapshots', t => {
  const f = fixture(t), group = f.group(), item = f.add(group), otherOwner = {};
  const hierarchy = new Map(), visibility = new Map();
  f.scene.traverse(object => { hierarchy.set(object, object.children.slice()); visibility.set(object, object.visible); });
  const project = (contextOwner = f.renderer) => captureThreeNodeClipping(T, {
    root: f.scene, camera: f.camera, hierarchy, visibility, contextOwner, renderContext: 'canvas',
    fail: (_, message) => assert.fail(message),
  }).get(item.object);
  const before = project(), other = project(otherOwner);
  const descriptor = f.programs.compile(item.material, item.object, f.context(item, before));
  const packet = f.capture(item, before, descriptor), bytes = packet.uniforms.map(value => value.slice());
  const planes = before.unionPlanes.map(plane => plane.toArray());
  assert.equal(project().cacheKey, before.cacheKey, 'live ownership retains the selected context');

  f.programs.dispose();
  const programs = createThreeNodePrograms({three: T, renderer: f.renderer});
  t.after(() => programs.dispose());
  const recreated = project();
  assert.notEqual(recreated.cacheKey, before.cacheKey, 'source RenderContexts disposal releases old group identities');
  assert.equal(project().cacheKey, recreated.cacheKey, 'recreated ownership retains its new context');
  assert.equal(project(otherOwner).cacheKey, other.cacheKey, 'another renderer keeps its own context');
  group.clippingPlanes[0].constant = .875; f.renderer.info.calls++;
  const after = project(), context = f.context(item, after);
  const next = programs.compile(item.material, item.object, context);
  const updated = programs.captureDraw(next, context);
  assert.notDeepEqual(updated.uniforms, bytes, 'new draws observe the new projected planes');
  assert.deepEqual(before.unionPlanes.map(plane => plane.toArray()), planes);
  assert.deepEqual(packet.uniforms, bytes, 'recreating ownership cannot rewrite accepted uniform packets');
});

for (const family of ['MeshBasic', 'MeshStandard', 'MeshPhysical'])
test(`ordinary ${family} custom keys compose with clipped source builders`, t => {
  const f = fixture(t), group = f.group(), material = new T[family + 'Material'](), calls = [];
  material.customProgramCacheKey = () => { calls.push('key'); return 'clipped-ordinary'; };
  const item = f.add(group, material), projected = f.clipping.getGroupContext(group);
  const source = f.objects.get(item.object, material, f.scene, f.camera, N.lights([]), {id: 11}, projected);
  source.getNodeBuilderState(); const expected = calls.splice(0);
  const descriptor = f.programs.compile(material, item.object, f.context(item, projected));
  assert.deepEqual(calls.splice(0), expected);
  assert.match(descriptor.fragment, /discard/);
  assert.deepEqual(unpack(descriptor, f.capture(item, projected, descriptor)), f.referenceDraw(source));
  material.customProgramCacheKey = () => { calls.push('replacement'); return 'replacement'; };
  assert.equal(f.programs.compile(material, item.object, f.context(item, projected)), descriptor);
  assert.deepEqual(calls, [], 'replacing the callback alone does not invalidate source construction');
  material.needsUpdate = true;
  const next = f.programs.compile(material, item.object, f.context(item, projected));
  assert.notEqual(next, descriptor); assert.deepEqual(calls, ['replacement', 'replacement']);
});

test('clipped extensions reject unowned plane keys and accessor metadata before authored construction', t => {
  const f = fixture(t), group = f.group(), item = f.add(group), projected = f.clipping.getGroupContext(group);
  let reads = 0;
  for (const key of [undefined, 123, {get() { reads++; return 'key'; }}]) {
    const clipping = {unionPlanes: projected.unionPlanes, intersectionPlanes: projected.intersectionPlanes};
    if (key?.get) Object.defineProperty(clipping, 'cacheKey', key);
    else if (key !== undefined) clipping.cacheKey = key;
    assert.throws(() => f.programs.compile(item.material, item.object, f.context(item, clipping)), {code: 'THREE_NODE_PROGRAM_CLIPPING'});
  }
  assert.equal(reads, 0); assert.deepEqual(f.trace, []); assert.equal(f.programs.diagnostics.programs, 0);
});

for (const registered of [false, true])
test(`material version short-circuit keeps its retained clipping arrays until source invalidation (${registered})`, t => {
  const f = fixture(t, 'MeshStandard', registered), left = f.group(), right = f.group(new T.Vector3(0, 1, 0), .625);
  const item = f.add(left), a = f.clipping.getGroupContext(left), b = f.clipping.getGroupContext(right);
  let first = f.compare(item, a);
  f.capture(item, a, first.descriptor); f.referenceDraw(first.source);
  left.clippingPlanes[0].constant = .875; f.clipping.getGroupContext(left);
  right.add(item.object); f.material.needsUpdate = true; f.renderer.info.calls++;
  const delayed = f.compare(item, b);
  assert.equal(delayed.reference, first.reference); assert.equal(delayed.descriptor, first.descriptor);
  assert.deepEqual(unpack(delayed.descriptor, f.capture(item, b, delayed.descriptor)), f.referenceDraw(delayed.source),
    'the retained builder still reads its original group, including that group\'s live coefficient update');
  f.renderer.info.calls++;
  let selected = f.compare(item, b);
  assert.notEqual(selected.reference, first.reference); assert.notEqual(selected.descriptor, first.descriptor);
  assert.deepEqual(unpack(selected.descriptor, f.capture(item, b, selected.descriptor)), f.referenceDraw(selected.source));
  for (const planes of [
    [new T.Plane(new T.Vector3(0, 1, 0), .375), new T.Plane(new T.Vector3(1, 0, 0), .125)],
    [],
  ]) {
    right.clippingPlanes = planes; f.clipping.getGroupContext(right);
    f.material.needsUpdate = true; f.renderer.info.calls++;
    const previous = selected; selected = f.compare(item, b);
    assert.equal(selected.reference, previous.reference); assert.equal(selected.descriptor, previous.descriptor);
    assert.deepEqual(unpack(selected.descriptor, f.capture(item, b, selected.descriptor)), f.referenceDraw(selected.source),
      'source fixed-size uniform allocation keeps truncation and previously uploaded tail behavior');
    f.renderer.info.calls++; selected = f.compare(item, b);
    assert.notEqual(selected.reference, previous.reference); assert.notEqual(selected.descriptor, previous.descriptor);
    assert.deepEqual(unpack(selected.descriptor, f.capture(item, b, selected.descriptor)), f.referenceDraw(selected.source));
  }
});

test('throwing clipping-key comparison preserves the source getter effect and retained plane owner', t => {
  const f = fixture(t), left = f.group(), right = f.group(new T.Vector3(0, 1, 0), .625), item = f.add(left);
  const a = f.clipping.getGroupContext(left), b = f.clipping.getGroupContext(right), first = f.compare(item, a);
  f.capture(item, a, first.descriptor); f.referenceDraw(first.source);
  const sentinel = new Error('application key'), key = item.material.customProgramCacheKey;
  let throwing = true, calls = 0;
  item.material.customProgramCacheKey = function () { calls++; if (throwing) throw sentinel; return key.call(this); };
  right.add(item.object); f.renderer.info.calls++;
  assert.throws(() => f.objects.get(item.object, item.material, f.scene, f.camera, f.lights, f.renderContext, b), error => error === sentinel);
  assert.throws(() => f.programs.compile(item.material, item.object, f.context(item, b)), error => error === sentinel);
  assert.equal(calls, 2); throwing = false;
  const retained = f.compare(item, b);
  assert.equal(retained.reference, first.reference); assert.equal(retained.descriptor, first.descriptor); assert.equal(calls, 2);
  assert.deepEqual(unpack(retained.descriptor, f.capture(item, b, retained.descriptor)), f.referenceDraw(retained.source));
});

test('a disposing key retains the source local clipped draw without resurrecting its cached owner', t => {
  const make = () => {
    const f = fixture(t), left = f.group(), right = f.group(new T.Vector3(0, 1, 0), .625), item = f.add(left);
    const a = f.clipping.getGroupContext(left), b = f.clipping.getGroupContext(right), key = f.material.customProgramCacheKey;
    let disposeNext = false;
    f.material.customProgramCacheKey = function () {
      if (disposeNext) { disposeNext = false; this.dispose(); }
      return key.call(this);
    };
    return {...f, item, a, b, change() {
      left.clippingPlanes[0].constant = .875; f.clipping.getGroupContext(left); right.add(item.object);
      f.material.needsUpdate = true; f.renderer.info.calls++; disposeNext = true;
    }};
  };
  const source = make(), candidate = make();
  const reference = clipping => source.objects.get(source.item.object, source.material, source.scene, source.camera,
    source.lights, source.renderContext, clipping);
  const compile = clipping => candidate.programs.compile(candidate.material, candidate.item.object, candidate.context(candidate.item, clipping));
  const initialSource = reference(source.a), initialBuilder = initialSource.getNodeBuilderState(), initial = compile(candidate.a);
  source.referenceDraw(initialSource); candidate.capture(candidate.item, candidate.a, initial);
  source.change(); candidate.change();
  const retainedSource = reference(source.b), retained = compile(candidate.b);
  assert.equal(retainedSource, initialSource); assert.equal(retainedSource.getNodeBuilderState(), initialBuilder);
  assert.equal(retained, initial); assert.equal(candidate.programs.diagnostics.programs, 0);
  assert.deepEqual(unpack(retained, candidate.capture(candidate.item, candidate.b, retained)), source.referenceDraw(retainedSource));
  const replacementSource = reference(source.b), replacement = compile(candidate.b);
  assert.notEqual(replacementSource, retainedSource); assert.notEqual(replacement, retained);
  assert.equal(candidate.programs.diagnostics.programs, 1);
  assert.deepEqual(unpack(replacement, candidate.capture(candidate.item, candidate.b, replacement)), source.referenceDraw(replacementSource));
});

test('public cold renders queue clipped material extensions with source projection before callbacks', async t => {
  const f = fixture(t, 'MeshBasic'), device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxBindingsPerBindGroup: 1000});
  device.createTexture = descriptor => ({dimension: '2d', depthOrArrayLayers: 1, sampleCount: 1, ...descriptor,
    width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
    createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }});
  device.createSampler = descriptor => ({...descriptor}); device.queue.writeTexture = () => {};
  let configuration;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) { configuration = value; }, unconfigure() {},
    getCurrentTexture() { return device.createTexture({size: [32, 32], format: configuration.format, usage: 16}); },
  })};
  const Renderer = createWebGPURendererClass(T, {maxDraws: 16}), renderer = new Renderer({canvas, device});
  renderer.outputColorSpace = T.LinearSRGBColorSpace; t.after(() => renderer.dispose());
  const left = f.group(), right = f.group(new T.Vector3(0, 1, 0), .625), a = f.add(left), b = f.add(right);
  a.object.renderOrder = 0; b.object.renderOrder = 1; a.object.frustumCulled = b.object.frustumCulled = false;
  let updates = 0;
  f.material.colorNode = N.uniform(new T.Color(.25, .5, .75)).onObjectUpdate(() => {
    updates++; left.clippingPlanes[0].constant += .125; return new T.Color(.25, .5, .75);
  });
  const captures = [], original = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function (request) {
    const value = original.call(this, request); if (value) captures.push(value); return value;
  };
  await renderer.init();
  renderer.render(f.scene, f.camera); renderer.render(f.scene, f.camera);
  while (renderer._drain) await renderer._drain;
  assert.equal(updates, 4, 'asynchronous preparation never repeats source draw callbacks');
  assert.equal(captures.length, 2); assert.equal(captures[0].draws.length, 2);
  const planes = capture => capture.draws.map(draw => {
    const descriptor = draw.binding.nodeProgram;
    const match = /@binding\(\s*(\d+)\s*\)\s*@group\(\s*(\d+)\s*\)\s*var<uniform>\s+NodeBuffer_\d+/.exec(descriptor.fragment);
    assert.ok(match, 'source clipping array is in the emitted shader');
    const group = descriptor.groups[Number(match[2])], member = group.uniforms.find(value => value.binding === Number(match[1]));
    const bytes = draw.nodeUniforms[group.index];
    return [...new Float32Array(bytes.buffer, bytes.byteOffset + member.offset, member.byteLength / 4)];
  });
  assert.deepEqual(planes(captures[0]), [[-1, -0, 0, .25], [-0, -1, 0, .625]]);
  assert.deepEqual(planes(captures[1]), [[-1, -0, 0, .5], [-0, -1, 0, .625]]);
  const draws = device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
  assert.equal(draws.length, 4, 'all accepted draws reach the owned GPU command queue');
  assert.ok(draws.every(draw => draw.pipeline.fragment.module.code.includes('discard')));
  const uniformBytes = draw => [...draw.groups.values()].flatMap(({group, offsets}) => {
    let dynamic = 0;
    return group.layout.entries.filter(entry => entry.buffer?.type === 'uniform').map(entry => {
      const resource = group.entries.find(value => value.binding === entry.binding).resource;
      const offset = (resource.offset ?? 0) + (entry.buffer.hasDynamicOffset ? offsets[dynamic++] ?? 0 : 0);
      return draw.contents.get(resource.buffer).slice(offset, offset + resource.size);
    });
  });
  for (const [index, draw] of draws.entries()) {
    const captured = captures[Math.floor(index / 2)].draws[index % 2];
    assert.deepEqual(uniformBytes(draw), unpack(captured.binding.nodeProgram, {uniforms: captured.nodeUniforms}),
      'the submitted bytes retain that queued draw\'s plane and material generation');
  }
});
