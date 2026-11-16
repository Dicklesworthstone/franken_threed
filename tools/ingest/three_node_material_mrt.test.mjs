/** Material extensions and MRT use the pinned source builder/cache lifecycle.
 * Source WGSL and recording-device checks do not execute GPU shader arithmetic. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import SourceRenderContexts from '../../upstream/three.js/src/renderers/common/RenderContexts.js';
import SourceRenderObjects from '../../upstream/three.js/src/renderers/common/RenderObjects.js';
import SourceNodeManager from '../../upstream/three.js/src/renderers/common/nodes/NodeManager.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL;
function target(t, {width = 16, height = 12, count = 2, type = T.HalfFloatType} = {}) {
  const output = new T.RenderTarget(width, height, {count, type});
  output.textures.forEach((texture, index) => { texture.name = ['output', 'normal', 'payload'][index]; });
  t.after(() => output.dispose());
  return output;
}
function materialExtension(renderer, {family = 'Basic', registered = false, trace = [], updates = []} = {}) {
  const name = ['Basic', 'Standard', 'Physical'].includes(family) ? `Mesh${family}` : family;
  const Native = T[`${name}NodeMaterial`], Ordinary = T[`${name}Material`];
  const pulse = () => N.uniform(.375).setName('mrtPulse').onObjectUpdate(frame => {
    updates.push([frame.object, frame.material]); return frame.material.applicationGain;
  });
  class Extension extends Native {
    constructor() { super(); this.applicationGain = .375; this.applicationVariant = 0; this.applicationPulse = pulse(); }
    customProgramCacheKey() { trace.push('key'); return super.customProgramCacheKey() + ':' + this.applicationVariant; }
    setupOutput(builder, output) {
      trace.push(['setup', builder.object, builder.material, builder.context.material]);
      if (output.isOutputStructNode === true) return super.setupOutput(builder, output);
      return super.setupOutput(builder, N.vec4(output.rgb.mul(this.applicationPulse), output.a));
    }
  }
  class Application extends Ordinary {
    constructor() {
      super(); this.type = `MRT${family}Application`; this.applicationGain = .375;
      this.applicationVariant = 0; this.applicationPulse = pulse();
    }
    customProgramCacheKey() { trace.push('key'); return super.customProgramCacheKey() + ':' + this.applicationVariant; }
  }
  if (registered) renderer.library.addMaterial(Extension, `MRT${family}Application`);
  return new (registered ? Application : Extension)();
}
function sourceFixture(t) {
  let selected = null, mrt = null;
  const renderer = {library: new T.StandardNodeLibrary(), contextNode: N.context(),
    debug: {diagnostics: {keywords: false}, onNodeBuilderCreated: null}, depth: true, lighting: {enabled: true},
    shadowMap: {enabled: false, type: T.PCFShadowMap}, _currentSourceMaterial: null,
    backend: {isWebGPUBackend: true, hasFeature: () => false,
      utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})}},
    hasFeature: () => false, getRenderTarget: () => selected, getOutputRenderTarget: () => null,
    getMRT: () => mrt, getClearDepth: () => 1, getClearStencil: () => 0,
    currentSamples: 1, _nodes: {nodeFrame: new T.NodeFrame()}, info: {calls: 1}};
  renderer.backend.createNodeBuilder = object => new T.WGSLNodeBuilder(object, renderer);
  const programs = createThreeNodePrograms({three: T, renderer}), contexts = new SourceRenderContexts(renderer);
  const nodes = new SourceNodeManager(renderer, renderer.backend);
  const source = new SourceRenderObjects(renderer, nodes, {}, {delete() {}}, {deleteForRender() {}}, {});
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 30), lights = N.lights([]);
  camera.position.z = 4; camera.coordinateSystem = T.WebGPUCoordinateSystem;
  camera.updateProjectionMatrix(); camera.updateMatrixWorld();
  t.after(() => { programs.dispose(); source.dispose(); contexts.dispose(); });
  const add = (material, object = new T.Mesh(new T.PlaneGeometry(), material)) => {
    scene.add(object); scene.updateMatrixWorld();
    object.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, object.matrixWorld);
    object.normalMatrix.getNormalMatrix(object.modelViewMatrix);
    t.after(() => { object.geometry.dispose(); material.dispose(); });
    return {object, material};
  };
  const context = item => ({...item, scene, camera, lights: [], sourceLightsNode: lights,
    renderId: renderer.info.calls, frameId: renderer.info.calls, colorAttachmentCount: selected?.textures.length ?? 1});
  const referenceObject = item => source.get(item.object, item.material, scene, camera, lights,
    contexts.get(selected, mrt), null);
  const reference = item => referenceObject(item).getNodeBuilderState();
  const referenceDraw = item => {
    const object = referenceObject(item);
    nodes.nodeFrame.frameId = nodes.nodeFrame.renderId = renderer.info.calls;
    nodes.updateBefore(object); nodes.updateForRender(object);
    const uniforms = object.getBindings().flatMap(group => group.bindings).filter(binding => binding.isUniformsGroup).map(binding => {
      binding.update(); return new Uint8Array(binding.buffer.buffer, binding.buffer.byteOffset, binding.buffer.byteLength).slice();
    });
    nodes.updateAfter(object); return uniforms;
  };
  return {renderer, programs, scene, camera, add, context, reference, referenceDraw,
    select(output, graph = mrt) { selected = output; mrt = graph; renderer.info.calls++; },
    compile: item => programs.compile(item.material, item.object, context(item)),
    capture: (item, descriptor, options = {}) => programs.captureDraw(descriptor, {...context(item), ...options})};
}
const words = packet => packet.uniforms.flatMap(bytes => [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);
const shadersEqual = (descriptor, reference) => {
  assert.equal(descriptor.vertex, reference.vertexShader); assert.equal(descriptor.fragment, reference.fragmentShader);
};

for (const [family, ObjectType, topology] of [
  ['Sprite', T.Sprite, 'triangles'], ['Points', T.Points, 'points'], ['Points', T.Sprite, 'triangles'],
  ['LineBasic', T.Line, 'line-strip'], ['LineBasic', T.LineSegments, 'lines'],
  ['LineDashed', T.Line, 'line-strip'], ['LineDashed', T.LineSegments, 'lines'],
]) for (const registered of [false, true])
test(`${registered ? 'registered' : 'inherited'} ${family} primitive MRT on ${ObjectType.name} matches source WGSL and uniforms`, t => {
  const f = sourceFixture(t), trace = [], updates = [];
  Object.assign(f.renderer, {getPixelRatio: () => 1.5,
    getSize: value => value.set(16, 12), getViewport: value => value.set(0, 0, 16, 12),
    getDrawingBufferSize: value => value.set(24, 18)});
  const material = materialExtension(f.renderer, {family, registered, trace, updates});
  material.color.setRGB(.25, .5, .75);
  if (family === 'Points') material.size = 13;
  if (family === 'LineDashed') { material.dashSize = .625; material.gapSize = .25; }
  let object;
  if (ObjectType === T.Sprite) { object = new T.Sprite(material); object.center.set(.125, .25); }
  else {
    const geometry = new T.BufferGeometry();
    geometry.setAttribute('position', new T.Float32BufferAttribute([-.5, -.25, 0, -.25, .5, 0, .25, -.5, 0, .5, .25, 0], 3));
    geometry.setAttribute('lineDistance', new T.Float32BufferAttribute([0, .25, .5, .75], 1));
    object = new ObjectType(geometry, material);
  }
  const item = f.add(material, object);
  f.select(target(t), N.mrt({output: N.output, normal: N.vec4(N.materialReference('applicationGain', 'float'))}));
  const check = () => {
    trace.length = 0;
    const reference = f.reference(item), expectedTrace = trace.splice(0), descriptor = f.compile(item);
    assert.deepEqual(trace.splice(0), expectedTrace); shadersEqual(descriptor, reference);
    assert.equal(descriptor.topology, topology); assert.equal(descriptor.colorCount, 2);
    assert.deepEqual(descriptor.colorTypes, ['vec4', 'vec4']);
    const expected = f.referenceDraw(item), expectedUpdates = updates.splice(0);
    const packet = f.capture(item, descriptor, {captureGeometry() {}});
    assert.deepEqual(packet.uniforms, expected); assert.deepEqual(updates.splice(0), expectedUpdates);
    return {reference, descriptor, packet};
  };
  const first = check(), saved = first.packet.uniforms.map(bytes => bytes.slice());
  assert.ok(words(first.packet).includes(.375));
  material.applicationGain = .875; f.renderer.info.calls++;
  const live = check(); assert.equal(live.descriptor, first.descriptor);
  assert.ok(words(live.packet).includes(.875)); assert.deepEqual(first.packet.uniforms, saved);
  material.applicationVariant++; material.needsUpdate = true; f.renderer.info.calls++;
  const keyed = check(); assert.notEqual(keyed.reference, first.reference); assert.notEqual(keyed.descriptor, first.descriptor);
});

for (const family of ['Basic', 'Standard', 'Physical']) for (const registered of [false, true])
test(`${registered ? 'registered' : 'inherited'} ${family} MRT materials preserve source graphs, uniforms and sharing`, t => {
  const f = sourceFixture(t), trace = [], updates = [];
  const material = materialExtension(f.renderer, {family, registered, trace, updates});
  const output = target(t), graph = N.mrt({output: N.output, normal: N.normalView});
  const a = f.add(material), b = f.add(material); f.select(output, graph); trace.length = 0;
  const check = item => {
    const reference = f.reference(item), expectedTrace = trace.splice(0), descriptor = f.compile(item);
    assert.deepEqual(trace.splice(0), expectedTrace); shadersEqual(descriptor, reference);
    assert.equal(descriptor.colorCount, 2); assert.deepEqual(descriptor.colorTypes, ['vec4', 'vec4']);
    const expected = f.referenceDraw(item), expectedUpdates = updates.splice(0), packet = f.capture(item, descriptor);
    assert.deepEqual(packet.uniforms, expected); assert.deepEqual(updates.splice(0), expectedUpdates);
    return {reference, descriptor, packet};
  };
  const first = check(a), other = check(b), saved = first.packet.uniforms.map(bytes => bytes.slice());
  assert.equal(first.reference, other.reference, 'source-equivalent objects share their MRT builder');
  material.applicationGain = .875; f.renderer.info.calls++;
  const changed = check(a); assert.equal(changed.descriptor, first.descriptor);
  assert.ok(words(changed.packet).includes(.875)); assert.deepEqual(first.packet.uniforms, saved);
  material.needsUpdate = true; f.renderer.info.calls++;
  assert.equal(check(a).descriptor, first.descriptor, 'value-only invalidation keeps the constructed source graph');
  material.applicationVariant = 1; material.needsUpdate = true; f.renderer.info.calls++;
  const keyed = check(a); assert.notEqual(keyed.reference, first.reference); assert.notEqual(keyed.descriptor, first.descriptor);
  material.dispose(); assert.equal(f.programs.diagnostics.programs, 0); f.renderer.info.calls++;
  assert.notEqual(check(a).reference, keyed.reference); assert.deepEqual(first.packet.uniforms, saved);
});

for (const registered of [false, true])
test(`${registered ? 'registered' : 'inherited'} MRT cache follows source context identity across targets and graph changes`, t => {
  const f = sourceFixture(t), trace = [], material = materialExtension(f.renderer, {registered, trace});
  const item = f.add(material), a = target(t), b = target(t, {width: 9, height: 7});
  const firstGain = N.uniform(.125).setName('firstMRTGain'), nextGain = N.uniform(.875).setName('nextMRTGain');
  const graphA = N.mrt({output: N.output, normal: N.vec4(firstGain)});
  const graphB = N.mrt({output: N.output, normal: N.vec4(nextGain)});
  const check = () => {
    trace.length = 0;
    const reference = f.reference(item), expected = trace.splice(0), descriptor = f.compile(item);
    assert.deepEqual(trace.splice(0), expected); shadersEqual(descriptor, reference);
    assert.deepEqual(f.capture(item, descriptor).uniforms, f.referenceDraw(item));
    return {reference, descriptor};
  };
  f.select(a, graphA); const first = check();
  f.select(b, graphA); const otherTarget = check();
  assert.equal(otherTarget.reference, first.reference); assert.equal(otherTarget.descriptor, first.descriptor);
  b.setSize(24, 10); f.renderer.info.calls++;
  assert.equal(check().descriptor, first.descriptor, 'source context identity excludes allocation size');
  f.select(b, graphB); const otherGraph = check();
  assert.notEqual(otherGraph.reference, first.reference); assert.notEqual(otherGraph.descriptor, first.descriptor);
  assert.match(otherGraph.descriptor.fragment, /nextMRTGain/);
  f.select(a, graphA); assert.equal(check().descriptor, first.descriptor);
  graphA.outputNodes.normal = N.vec4(nextGain); graphA.needsUpdate = true; f.renderer.info.calls++;
  const staleGraph = check();
  assert.equal(staleGraph.reference, first.reference); assert.equal(staleGraph.descriptor, first.descriptor);
  assert.match(staleGraph.descriptor.fragment, /firstMRTGain/);
  assert.doesNotMatch(staleGraph.descriptor.fragment, /nextMRTGain/);
  material.needsUpdate = true; f.renderer.info.calls++;
  assert.equal(check().descriptor, first.descriptor, 'renderer MRT contents are not a source material cache-key dimension');
  material.dispose(); f.renderer.info.calls++;
  const rebuilt = check(); assert.notEqual(rebuilt.reference, first.reference);
  assert.match(rebuilt.descriptor.fragment, /nextMRTGain/);
  f.select(null, graphA);
  const canvas = check(); assert.equal(canvas.descriptor.colorCount, 1);
  assert.doesNotMatch(canvas.descriptor.fragment, /nextMRTGain/, 'source MRT outputs are inactive on the canvas');
  f.select(a, graphA); assert.equal(check().descriptor, rebuilt.descriptor);
});

for (const registered of [false, true])
test(`${registered ? 'registered' : 'inherited'} material MRT overrides retain their source invalidation boundary`, t => {
  const f = sourceFixture(t), trace = [], material = materialExtension(f.renderer, {registered, trace});
  const original = N.uniform(.25).setName('materialMRTOriginal'), replacement = N.uniform(.75).setName('materialMRTReplacement');
  material.mrtNode = N.mrt({normal: N.vec4(original)});
  const item = f.add(material), output = target(t), graph = N.mrt({output: N.output, normal: N.normalView});
  f.select(output, graph);
  const check = () => {
    trace.length = 0;
    const reference = f.reference(item), expected = trace.splice(0), descriptor = f.compile(item);
    assert.deepEqual(trace.splice(0), expected); shadersEqual(descriptor, reference);
    const packet = f.capture(item, descriptor); assert.deepEqual(packet.uniforms, f.referenceDraw(item));
    return {reference, descriptor, packet};
  };
  const first = check(); assert.match(first.descriptor.fragment, /materialMRTOriginal/);
  material.mrtNode.outputNodes.normal = N.vec4(replacement); material.mrtNode.needsUpdate = true; f.renderer.info.calls++;
  assert.equal(check().descriptor, first.descriptor, 'node mutation alone leaves the existing RenderObject intact');
  material.needsUpdate = true; f.renderer.info.calls++;
  if (registered) {
    assert.equal(check().descriptor, first.descriptor, 'ordinary Material keys do not recursively key a custom mrtNode');
    material.applicationVariant++; material.needsUpdate = true; f.renderer.info.calls++;
  }
  const changed = check(); assert.notEqual(changed.reference, first.reference);
  assert.match(changed.descriptor.fragment, /materialMRTReplacement/); assert.ok(words(changed.packet).includes(.75));
});

test('MRT changes made by node updates still reject before a draw packet is accepted', t => {
  const f = sourceFixture(t), material = materialExtension(f.renderer), item = f.add(material), output = target(t);
  const graph = N.mrt({output: N.output, normal: N.normalView}); f.select(output, graph);
  const mutate = N.uniform(1).onObjectUpdate(() => {
    graph.outputNodes.normal = N.vec4(.75); graph.needsUpdate = true; return 1;
  });
  material.opacityNode = mutate;
  const descriptor = f.compile(item);
  assert.throws(() => f.capture(item, descriptor), {code: 'THREE_NODE_PROGRAM_CHANGED'});
});

for (const kind of ['material-only MRT', 'typed fragment outputs'])
test(`${kind} on custom materials retains exact source output types without renderer MRT state`, t => {
  const f = sourceFixture(t), material = materialExtension(f.renderer), item = f.add(material), output = target(t);
  if (kind === 'material-only MRT') material.mrtNode = N.mrt({output: N.output, normal: N.vec4(.25, .5, .75, 1)});
  else {
    Object.assign(output.textures[1], {type: T.UnsignedIntType, format: T.RGBAIntegerFormat});
    material.fragmentNode = N.outputStruct(N.vec4(.25, .5, .75, 1), N.uvec4(16777217, 4294967295, 2147483648, 1));
  }
  f.select(output, null);
  const reference = f.reference(item), descriptor = f.compile(item);
  shadersEqual(descriptor, reference);
  assert.deepEqual(descriptor.colorTypes, ['vec4', kind === 'material-only MRT' ? 'vec4' : 'uvec4']);
  assert.deepEqual(f.capture(item, descriptor).uniforms, f.referenceDraw(item));
});

function nativeFixture(t) {
  const device = geometryDevice();
  Object.assign(device.limits, {maxTextureDimension2D: 8192, maxTextureDimension3D: 256, maxTextureArrayLayers: 256,
    maxColorAttachments: 8, maxColorAttachmentBytesPerSample: 64, maxBindingsPerBindGroup: 1000});
  device.features = new Set(); device.textures = [];
  device.createTexture = descriptor => {
    const texture = {dimension: '2d', sampleCount: 1, ...descriptor,
      width: descriptor.size.width ?? descriptor.size[0], height: descriptor.size.height ?? descriptor.size[1],
      depthOrArrayLayers: descriptor.size.depthOrArrayLayers ?? descriptor.size[2] ?? 1,
      createView(options = {}) { return {texture: this, ...options}; }, destroy() { this.destroyed = true; }};
    device.textures.push(texture); return texture;
  };
  device.createSampler = descriptor => ({...descriptor}); device.queue.writeTexture = () => {};
  device.createRenderPipeline = descriptor => {
    const pipeline = {...descriptor, getBindGroupLayout: () => ({})}; device.pipelines.push(pipeline); return pipeline;
  };
  let configured, acquired = 0;
  const canvas = {width: 32, height: 32, style: {}, getContext: () => ({
    configure(value) { configured = value; }, unconfigure() {},
    getCurrentTexture() { acquired++; return device.createTexture({size: [32, 32], format: configured.format, usage: 16}); },
  })};
  const Renderer = createWebGPURendererClass(T), renderer = new Renderer({canvas, device});
  renderer.outputColorSpace = T.LinearSRGBColorSpace; t.after(() => renderer.dispose());
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 30); camera.position.z = 4;
  const captures = [], capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function(request) {
    const value = capture.call(this, request); if (value) captures.push(value); return value;
  };
  return {renderer, device, scene, camera, captures, acquired: () => acquired};
}
const nativeDraws = device => device.snapshots.flat().filter(draw => draw.pipeline.vertex.entryPoint === 'main');
function drawUniforms(draw, Type) {
  return [...draw.groups.values()].flatMap(({group, offsets}) => group.layout.entries
    .filter(entry => entry.buffer?.type === 'uniform').sort((a, b) => a.binding - b.binding)
    .flatMap((entry, index) => {
      const resource = group.entries.find(value => value.binding === entry.binding).resource;
      const bytes = draw.contents.get(resource.buffer), offset = (resource.offset ?? 0) + (offsets[index] ?? 0);
      return [...new Type(bytes.buffer, bytes.byteOffset + offset, resource.size / 4)];
    }));
}

for (const registered of [false, true])
test(`${registered ? 'registered' : 'inherited'} native MRT draws capture queued mixed attachments and immutable custom uniforms`, async t => {
  const f = nativeFixture(t), updates = [], material = materialExtension(f.renderer, {registered, updates});
  const object = new T.Mesh(new T.PlaneGeometry(), material); object.frustumCulled = false; f.scene.add(object);
  t.after(() => { object.geometry.dispose(); material.dispose(); });
  material.applicationPayload = 16777217;
  const a = target(t, {count: 3}), b = target(t, {count: 3, width: 9, height: 7});
  for (const output of [a, b]) Object.assign(output.textures[2], {type: T.UnsignedIntType, format: T.RGBAIntegerFormat});
  const graph = N.mrt({output: N.output, normal: N.normalView,
    payload: N.uvec4(N.materialReference('applicationPayload', 'uint'))});
  f.renderer.setMRT(graph); f.renderer.setRenderTarget(a); await f.renderer.init();
  const first = f.renderer.renderAsync(f.scene, f.camera);
  material.applicationGain = .875; material.applicationPayload = 4294967295;
  f.renderer.setRenderTarget(b); const second = f.renderer.renderAsync(f.scene, f.camera);
  material.applicationGain = .625; material.applicationPayload = 2147483648;
  f.renderer.setRenderTarget(a); const third = f.renderer.renderAsync(f.scene, f.camera);
  material.applicationGain = .9375; material.applicationPayload = 1;
  await Promise.all([first, second, third]);
  a.setSize(24, 10); await f.renderer.renderAsync(f.scene, f.camera);
  const draws = nativeDraws(f.device), passes = f.device.passes.filter(pass => pass.draws.length);
  assert.equal(draws.length, 4); assert.equal(passes.length, 4); assert.equal(f.acquired(), 0);
  const descriptor = f.captures[0].draws[0].binding.nodeProgram;
  assert.deepEqual(descriptor.colorTypes, ['vec4', 'vec4', 'uvec4']);
  for (let index = 0; index < 4; index++) {
    assert.equal(f.captures[index].draws[0].binding.nodeProgram, descriptor);
    assert.deepEqual(draws[index].pipeline.fragment.targets.map(entry => entry.format), ['rgba16float', 'rgba16float', 'rgba32uint']);
    assert.equal(draws[index].pipeline.fragment.targets[2].blend, undefined);
    assert.ok(drawUniforms(draws[index], Float32Array).includes([.375, .875, .625, .9375][index]));
    assert.ok(drawUniforms(draws[index], Uint32Array).includes([16777217, 4294967295, 2147483648, 1][index]));
    const attachments = passes[index].desc.colorAttachments.map(attachment => attachment.view.texture);
    assert.equal(new Set(attachments).size, 3);
    assert.deepEqual(attachments.map(texture => [texture.width, texture.height]),
      Array.from({length: 3}, () => [[16, 12], [9, 7], [16, 12], [24, 10]][index]));
  }
  assert.equal(updates.length, 4); for (const identities of updates) assert.deepEqual(identities, [object, material]);
  assert.equal(f.renderer._nodeCaptureBytes, 0); assert.equal(f.renderer.info.f3d.route, 'general-webgpu');
});
