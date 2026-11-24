import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/src/Three.WebGPU.js';
import SourceRenderObjects from '../../upstream/three.js/src/renderers/common/RenderObjects.js';
import SourceNodeManager from '../../upstream/three.js/src/renderers/common/nodes/NodeManager.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {createWebGPURendererClass} from './three_renderer.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';

const N = T.TSL;
const canonical = shader => shader.replace(/NodeBuffer_\d+/g, 'NodeBuffer');
const unpack = (descriptor, packet) => descriptor.groups.flatMap((group, index) =>
  group.uniforms.map(member => packet.uniforms[index].slice(member.offset, member.offset + member.byteLength)));
const values = packet => packet.uniforms.flatMap(bytes => [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);

function fixture(t, family = 'MeshStandard', ownership = 'inherited') {
  const trace = [];
  class Extension extends T[family + 'NodeMaterial'] {
    constructor() { super(); this.applicationVariant = 0; }
    customProgramCacheKey() {
      trace.push('key'); this.userData.onKey?.();
      return super.customProgramCacheKey() + ':' + this.applicationVariant;
    }
    setupOutput(builder, output) {
      trace.push('setup'); this.userData.onSetup?.(builder);
      return super.setupOutput(builder, output);
    }
  }
  class Application extends T[family + 'Material'] {
    constructor() { super(); this.type = 'ContextApplication'; this.applicationVariant = 0; }
    customProgramCacheKey() {
      trace.push('key'); this.userData.onKey?.();
      return super.customProgramCacheKey() + ':' + this.applicationVariant;
    }
  }
  const renderer = {library: new T.StandardNodeLibrary(), contextNode: N.context(),
    debug: {diagnostics: {keywords: false}, onNodeBuilderCreated: null}, depth: true, lighting: {enabled: true},
    shadowMap: {enabled: false, type: T.PCFShadowMap}, _currentSourceMaterial: null,
    backend: {isWebGPUBackend: true, hasFeature: () => false,
      utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})}},
    hasFeature: () => false, getRenderTarget: () => null, getOutputRenderTarget: () => null,
    getMRT: () => null, currentSamples: 1, _nodes: {nodeFrame: new T.NodeFrame()}, info: {calls: 1}};
  Object.defineProperty(renderer, 'highPrecision', Object.getOwnPropertyDescriptor(T.Renderer.prototype, 'highPrecision'));
  renderer.backend.createNodeBuilder = object => new T.WGSLNodeBuilder(object, renderer);
  if (ownership === 'registered') renderer.library.addMaterial(Extension, 'ContextApplication');
  const material = new (ownership === 'registered' ? Application : ownership === 'custom-key' ? T[family + 'Material'] : Extension)();
  if (ownership === 'custom-key') {
    material.applicationVariant = 0;
    material.customProgramCacheKey = function () {
      trace.push('key'); this.userData.onKey?.();
      return T.Material.prototype.customProgramCacheKey.call(this) + ':' + this.applicationVariant;
    };
  }
  const programs = createThreeNodePrograms({three: T, renderer});
  const nodes = new SourceNodeManager(renderer, renderer.backend);
  const objects = new SourceRenderObjects(renderer, nodes, {}, {delete() {}}, {deleteForRender() {}}, {});
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 30), geometry = new T.PlaneGeometry();
  camera.position.z = 4; camera.coordinateSystem = T.WebGPUCoordinateSystem;
  camera.updateProjectionMatrix(); camera.updateMatrixWorld();
  const lights = N.lights([]), renderContext = {id: 0};
  const matrices = () => {
    scene.updateMatrixWorld(); camera.updateMatrixWorld();
    for (const object of scene.children) {
      object.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, object.matrixWorld);
      object.normalMatrix.getNormalMatrix(object.modelViewMatrix);
    }
  };
  const add = () => { const object = new T.Mesh(geometry, material); scene.add(object); matrices(); return {object, material}; };
  const context = item => ({...item, scene, camera, lights: [], sourceLightsNode: lights,
    renderId: renderer.info.calls, frameId: renderer.info.calls});
  const reference = item => {
    const source = objects.get(item.object, material, scene, camera, lights, renderContext, null);
    return {source, reference: source.getNodeBuilderState()};
  };
  const compile = item => programs.compile(material, item.object, context(item));
  const compare = item => {
    const source = reference(item), expected = trace.splice(0), descriptor = compile(item);
    assert.deepEqual(trace.splice(0), expected, 'source keys, material setup and context hooks keep their order');
    assert.equal(canonical(descriptor.vertex), canonical(source.reference.vertexShader));
    assert.equal(canonical(descriptor.fragment), canonical(source.reference.fragmentShader));
    return {...source, descriptor};
  };
  const referenceDraw = source => {
    nodes.nodeFrame.frameId = nodes.nodeFrame.renderId = renderer.info.calls;
    nodes.updateBefore(source); nodes.updateForRender(source);
    const uniforms = source.getBindings().flatMap(group => group.bindings).filter(binding => binding.isUniformBuffer).map(binding => {
      binding.update(); const bytes = new Uint8Array(binding.byteLength), array = binding.buffer;
      bytes.set(new Uint8Array(array.buffer, array.byteOffset, array.byteLength)); return bytes;
    });
    nodes.updateAfter(source); return uniforms;
  };
  const capture = (item, selected) => {
    const packet = programs.captureDraw(selected.descriptor, context(item));
    assert.deepEqual(unpack(selected.descriptor, packet), referenceDraw(selected.source)); return packet;
  };
  t.after(() => { programs.dispose(); objects.dispose(); geometry.dispose(); material.dispose(); });
  return {trace, renderer, material, programs, nodes, objects, scene, camera, lights, renderContext, matrices, add, context,
    reference, referenceDraw, compile, compare, capture};
}

for (const ownership of ['inherited', 'registered', 'custom-key'])
for (const family of ['MeshBasic', 'MeshToon', 'MeshLambert', 'MeshPhong', 'MeshStandard', 'MeshPhysical'])
test(`${ownership} ${family} materials compose source AO/GI contexts and shared live uniforms`, t => {
  const f = fixture(t, family, ownership), a = f.add(), b = f.add();
  const ao = N.uniform(.375).setName('contextOcclusion'), gi = N.uniform(new T.Vector3(.25, .5, .75)).setName('contextIrradiance');
  f.renderer.contextNode = N.builtinGIContext(ao, gi);
  const hooks = f.renderer.contextNode.value;
  for (const key of ['getAO', 'getGI']) {
    const original = hooks[key]; hooks[key] = (...args) => { f.trace.push(key); return original(...args); };
  }
  f.programs.inspect(f.material, a.object, f.context(a)); assert.deepEqual(f.trace, [], 'admission does not execute context hooks');
  a.object.position.x = 2; b.object.position.x = 7; f.matrices();
  const first = f.compare(a), shared = f.compare(b);
  assert.equal(first.reference, shared.reference); assert.notEqual(first.descriptor, shared.descriptor);
  const accepted = f.capture(a, first), saved = accepted.uniforms.map(bytes => bytes.slice()); f.capture(b, shared);
  ao.value = .875; gi.value.set(.125, .625, .875); f.renderer.info.calls++;
  assert.equal(f.compare(a).descriptor, first.descriptor, 'live uniforms retain source construction');
  assert.notDeepEqual(f.capture(a, first).uniforms, accepted.uniforms);
  assert.deepEqual(accepted.uniforms, saved);
  f.material.needsUpdate = true;
  assert.equal(f.compare(a).descriptor, first.descriptor, 'version-only unchanged material keys retain context construction');
  f.renderer.contextNode.needsUpdate = true;
  const rebuilt = f.compare(a);
  assert.notEqual(rebuilt.reference, first.reference, 'root context invalidation is live within the same renderer call');
  assert.notEqual(rebuilt.descriptor, first.descriptor); f.capture(a, rebuilt);
  f.renderer.contextNode = N.builtinAOContext(ao);
  const replaced = f.compare(a);
  assert.notEqual(replaced.reference, rebuilt.reference); assert.notEqual(replaced.descriptor, rebuilt.descriptor);
  f.capture(a, replaced);
});

for (const ownership of ['inherited', 'registered', 'custom-key'])
test(`${ownership} renderer texture contexts retain nested precedence and unversioned shared hooks`, t => {
  const f = fixture(t, 'MeshBasic', ownership), a = f.add();
  const texture = new T.DataTexture(new Uint8Array(16).fill(255), 2, 2);
  texture.minFilter = texture.magFilter = T.LinearFilter; texture.needsUpdate = true;
  f.material.map = texture; t.after(() => texture.dispose());
  const uv = N.uniform(new T.Vector2(.125, .375)).setName('originalContextUV');
  const replacement = N.uniform(new T.Vector2(.625, .875)).setName('replacementContextUV');
  const child = N.context({getUV: () => { f.trace.push('child-uv'); return uv; }});
  f.renderer.contextNode = N.context(child, {getUV: () => { f.trace.push('parent-uv'); return N.vec2(0); },
    getTextureLevel: () => { f.trace.push('level'); return N.float(0); }, forceUVContext: true});
  const first = f.compare(a); assert.match(first.descriptor.fragment, /originalContextUV/);
  assert.match(first.descriptor.fragment, /textureSampleLevel/);
  const accepted = f.capture(a, first), saved = accepted.uniforms.map(bytes => bytes.slice());
  child.value.getUV = () => { f.trace.push('replacement-uv'); return replacement; };
  child.needsUpdate = true; f.renderer.info.calls++;
  assert.equal(f.compare(a).descriptor, first.descriptor, 'nested versions do not invalidate the root source key');
  const b = f.add(), cold = f.compare(b);
  assert.equal(cold.reference, first.reference, 'a cold object reuses the existing source builder and its original context');
  assert.match(cold.descriptor.fragment, /originalContextUV/); f.capture(b, cold);
  uv.value.set(.25, .5); f.renderer.info.calls++;
  assert.ok(values(f.capture(a, first)).includes(.25)); assert.deepEqual(accepted.uniforms, saved);
  f.material.needsUpdate = true;
  assert.equal(f.compare(a).descriptor, first.descriptor, 'material version alone retains unversioned context changes');
  f.renderer.contextNode.needsUpdate = true;
  const rebuilt = f.compare(a); assert.match(rebuilt.descriptor.fragment, /replacementContextUV/);
  assert.doesNotMatch(rebuilt.descriptor.fragment, /originalContextUV/); f.capture(a, rebuilt);
  f.material.applicationVariant++; f.material.needsUpdate = true;
  assert.notEqual(f.compare(a).descriptor, rebuilt.descriptor);
});

for (const ownership of ['inherited', 'registered', 'custom-key'])
test(`${ownership} highPrecision contexts retain source matrix lifetimes at large coordinates`, t => {
  const f = fixture(t, 'MeshPhong', ownership), item = f.add(), origin = 2 ** 28;
  f.camera.position.set(origin + .25, -origin + .5, origin + 4);
  item.object.position.set(origin + .75, -origin + .125, origin - 1.5); f.matrices();
  f.renderer.highPrecision = true;
  const first = f.compare(item); assert.match(first.descriptor.vertex, /highpModelViewMatrix/);
  const accepted = f.capture(item, first), saved = accepted.uniforms.map(bytes => bytes.slice());
  assert.ok(values(accepted).includes(-5.5)); assert.ok(values(accepted).includes(-.375));
  item.object.position.x += .375; f.camera.position.y += .25; f.matrices(); f.renderer.info.calls++;
  assert.equal(f.compare(item).descriptor, first.descriptor); f.capture(item, first);
  f.renderer.highPrecision = false;
  assert.equal(f.compare(item).descriptor, first.descriptor, 'the setter does not invalidate an existing source graph');
  f.capture(item, first);
  f.material.needsUpdate = true;
  assert.equal(f.compare(item).descriptor, first.descriptor);
  f.renderer.contextNode.needsUpdate = true;
  const ordinary = f.compare(item); assert.doesNotMatch(ordinary.descriptor.vertex, /highpModelViewMatrix/);
  f.capture(item, ordinary); assert.deepEqual(accepted.uniforms, saved);
  f.renderer.highPrecision = true; f.renderer.contextNode.needsUpdate = true;
  const restored = f.compare(item); assert.match(restored.descriptor.vertex, /highpModelViewMatrix/); f.capture(item, restored);
});

test('context admission retains unsupported-field, accessor and material-context boundaries before callbacks', t => {
  const f = fixture(t), item = f.add(); let reads = 0;
  for (const data of [{getOutput: output => output}, {forceUVContext: 1}, {getAO: N.float(1)},
    Object.defineProperty({}, 'getAO', {enumerable: true, get() { reads++; return () => N.float(1); }})]) {
    f.renderer.contextNode = N.context(data);
    assert.throws(() => f.compile(item), {code: 'THREE_NODE_PROGRAM_GRAPH'});
  }
  f.renderer.contextNode = N.builtinAOContext(N.float(.5));
  f.material.contextNode = N.builtinAOContext(N.float(.75));
  assert.throws(() => f.compile(item), {code: 'THREE_NODE_PROGRAM_HOOK'});
  assert.equal(reads, 0); assert.deepEqual(f.trace, []); assert.equal(f.programs.diagnostics.programs, 0);
});

for (const phase of ['key', 'setup', 'context', 'before', 'update'])
for (const transition of ['version', 'replacement'])
test(`renderer context ${transition} during ${phase} preserves source construction and update timing`, t => {
  const run = reference => {
    const f = fixture(t), item = f.add(), original = N.uniform(.375).setName('initialContextAO');
    const replacement = N.uniform(.875).setName('nextContextAO'); let changed = false;
    const change = at => {
      if (changed || at !== phase) return;
      changed = true; f.trace.push('change:' + at);
      if (transition === 'version') f.renderer.contextNode.needsUpdate = true;
      else f.renderer.contextNode = N.builtinAOContext(replacement);
    };
    f.renderer.contextNode = N.context({getAO() { f.trace.push('ao'); change('context'); return original; }});
    f.material.userData.onKey = () => change('key');
    f.material.userData.onSetup = () => change('setup');
    original.updateBeforeType = 'object';
    original.updateBefore = frame => {
      assert.equal(frame.object, item.object); assert.equal(frame.material, f.material);
      f.trace.push('before'); change('before');
    };
    original.onObjectUpdate(frame => {
      assert.equal(frame.object, item.object); assert.equal(frame.renderer, f.renderer);
      f.trace.push('update'); change('update'); return original.value;
    });
    const snapshots = [], selected = [], shaders = [];
    for (let i = 0; i < 2; i++) {
      f.renderer.info.calls++;
      if (reference) {
        const result = f.reference(item); selected.push(result.reference);
        shaders.push([canonical(result.reference.vertexShader), canonical(result.reference.fragmentShader)]);
        snapshots.push(f.referenceDraw(result.source));
      } else {
        const descriptor = f.compile(item); selected.push(descriptor);
        shaders.push([canonical(descriptor.vertex), canonical(descriptor.fragment)]);
        // Public scene capture separates updateBefore from uniform capture.
        f.programs.before(descriptor, f.context(item));
        const packet = f.programs.captureDraw(descriptor, {...f.context(item), beforeDone: true});
        f.programs.after(descriptor, f.context(item)); snapshots.push(unpack(descriptor, packet));
      }
    }
    assert.equal(changed, true); return {trace: f.trace, shaders, snapshots, rebuilt: selected[0] !== selected[1]};
  };
  const expected = run(true), actual = run(false);
  assert.deepEqual(actual, expected, 'the current draw retains its selected graph and the following draw observes the source root key');
  assert.equal(actual.rebuilt, phase !== 'key', 'a key callback already contributes its new context to the initial cache key');
});

test('renderer context changes remain visible when a custom key comparison throws or disposes its owner', t => {
  for (const action of ['throw', 'dispose']) {
    const run = reference => {
      const f = fixture(t), item = f.add(), first = N.uniform(.375).setName('oldContextAO');
      f.renderer.contextNode = N.builtinAOContext(first);
      const select = () => reference ? f.reference(item) : {descriptor: f.compile(item)};
      const initial = select(); let changed = false;
      f.material.userData.onKey = () => {
        if (changed) return;
        changed = true; f.trace.push(action);
        f.renderer.contextNode.value.getAO = () => N.uniform(.875).setName('changedContextAO');
        if (action === 'throw') throw new Error('application key failure');
        f.material.dispose();
      };
      f.material.needsUpdate = true;
      if (action === 'throw') assert.throws(select, /application key failure/);
      const selected = select(), source = selected.reference ?? selected.descriptor;
      const uniforms = reference ? f.referenceDraw(selected.source) :
        unpack(selected.descriptor, f.programs.captureDraw(selected.descriptor, f.context(item)));
      return {trace: f.trace, vertex: canonical(source.vertexShader ?? source.vertex),
        fragment: canonical(source.fragmentShader ?? source.fragment), uniforms,
        retained: (initial.reference ?? initial.descriptor) === source};
    };
    assert.deepEqual(run(false), run(true), 'failed comparisons and self-disposal keep their source-visible effects');
  }
});

test('an accepted empty renderer context still checks its live root before node callbacks', t => {
  const f = fixture(t), item = f.add(); let updates = 0;
  f.material.colorNode = N.uniform(.5).onObjectUpdate(() => { updates++; return .5; });
  const first = f.compare(item);
  f.renderer.contextNode = N.context();
  assert.throws(() => f.programs.captureDraw(first.descriptor, f.context(item)), {code: 'THREE_NODE_PROGRAM_CHANGED'});
  assert.equal(updates, 0);
});

for (const registered of [false, true])
test(`extension type history preserves an ordinary same-family node material (${registered})`, t => {
  const f = fixture(t, 'MeshBasic', registered ? 'registered' : 'inherited'), item = f.add();
  if (registered) {
    f.renderer.library.addMaterial(f.renderer.library.getMaterialNodeClass('ContextApplication'), 'MeshBasicNodeMaterial');
    f.material.type = 'MeshBasicNodeMaterial';
  }
  f.renderer.contextNode = N.builtinAOContext(N.uniform(.375));
  const ordinary = new T.MeshBasicNodeMaterial(); ordinary.colorNode = N.vec3(.25, .5, .75);
  const object = new T.Mesh(item.object.geometry, ordinary); f.scene.add(object); f.matrices();
  t.after(() => ordinary.dispose());
  const context = f.context({object, material: ordinary});
  const reference = () => f.objects.get(object, ordinary, f.scene, f.camera, f.lights, f.renderContext, null).getNodeBuilderState();
  const original = reference(), descriptor = f.programs.compile(ordinary, object, context);
  assert.equal(canonical(descriptor.fragment), canonical(original.fragmentShader));
  f.compare(item);
  assert.equal(reference(), original, 'source native NodeMaterials bypass ordinary material conversion');
  assert.equal(f.programs.compile(ordinary, object, context), descriptor,
    'a subclass or registered type cannot promote an unrelated native material to extension ownership');
});

for (const registered of [false, true])
test(`public native scene passes queue custom material contexts and immutable uniform generations (${registered})`, async t => {
  const device = geometryDevice();
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
  const Renderer = createWebGPURendererClass(T), renderer = new Renderer({canvas, device});
  renderer.outputColorSpace = T.LinearSRGBColorSpace;
  class Extension extends T.MeshBasicNodeMaterial {
    setupOutput(builder, output) { return super.setupOutput(builder, output); }
  }
  class Application extends T.MeshBasicMaterial { constructor() { super(); this.type = 'PassContextApplication'; } }
  if (registered) renderer.library.addMaterial(Extension, 'PassContextApplication');
  const scene = new T.Scene(), outer = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 30);
  camera.position.z = 4;
  const material = new (registered ? Application : Extension)(), geometry = new T.PlaneGeometry();
  const object = new T.Mesh(geometry, material); scene.add(object);
  const global = N.builtinAOContext(N.uniform(.125).setName('globalContextAO'));
  renderer.contextNode = global;
  const ao = N.uniform(.375).setName('passContextAO'), next = N.uniform(.5).setName('replacementPassAO');
  const pass = N.pass(scene, camera); pass.contextNode = N.builtinAOContext(ao);
  const output = new T.MeshBasicNodeMaterial(); output.colorNode = pass.getTextureNode();
  outer.add(new T.Mesh(geometry, output));
  t.after(() => { renderer.dispose(); pass.dispose(); geometry.dispose(); material.dispose(); output.dispose(); });
  const captured = [], capture = renderer._captureNodeFrame;
  renderer._captureNodeFrame = function (request) {
    const value = capture.call(this, request);
    for (const draw of value?.draws ?? []) if (draw.object === object) captured.push(draw);
    return value;
  };
  await renderer.init();
  const render = () => { renderer._advanceAnimationFrame(); renderer.render(outer, camera); assert.equal(renderer.contextNode, global); };
  render(); ao.value = .875; render();
  pass.contextNode = N.builtinAOContext(next); render();
  pass.needsUpdate = true; render();
  const accepted = captured.map(draw => draw.nodeUniforms.map(bytes => bytes.slice()));
  ao.value = next.value = 99;
  while (renderer._drain) await renderer._drain;
  assert.equal(captured.length, 4);
  const draws = device.snapshots.flat().filter(draw => /passContextAO|replacementPassAO/.test(draw.pipeline.fragment.module.code));
  assert.equal(draws.length, 4, 'every accepted scene-pass draw reaches the owned native queue');
  const uniformBytes = draw => [...draw.groups.values()].flatMap(({group, offsets}) => {
    let dynamic = 0;
    return group.layout.entries.filter(entry => entry.buffer?.type === 'uniform').map(entry => {
      const resource = group.entries.find(value => value.binding === entry.binding).resource;
      const offset = (resource.offset ?? 0) + (entry.buffer.hasDynamicOffset ? offsets[dynamic++] ?? 0 : 0);
      return draw.contents.get(resource.buffer).slice(offset, offset + resource.size);
    });
  });
  for (const [index, value] of [.375, .875, .875, .5].entries()) {
    const draw = captured[index], packet = {uniforms: draw.nodeUniforms};
    assert.ok(values(packet).includes(value)); assert.ok(!values(packet).includes(99));
    assert.deepEqual(draw.nodeUniforms, accepted[index]);
    assert.deepEqual(uniformBytes(draws[index]), unpack(draw.binding.nodeProgram, packet));
  }
  assert.match(draws[2].pipeline.fragment.module.code, /passContextAO/);
  assert.match(draws[3].pipeline.fragment.module.code, /replacementPassAO/);
  assert.equal(renderer._nodeCaptureBytes, 0);
});
