import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import SourceRenderObjects from '../../upstream/three.js/src/renderers/common/RenderObjects.js';
import SourceNodeManager from '../../upstream/three.js/src/renderers/common/nodes/NodeManager.js';
import SourceRenderer from '../../upstream/three.js/src/renderers/common/Renderer.js';
import {createThreeNodePrograms, hasThreeNativeMaterialExtension, threeNativeMaterialExtensionKey} from './three_node_program.mjs';

const N = T.TSL;
function fixture(t, options = {}) {
  const renderer = {library: new T.StandardNodeLibrary(), contextNode: N.context(),
    debug: {diagnostics: {keywords: false}, onNodeBuilderCreated: null}, depth: true, lighting: {enabled: true},
    shadowMap: {enabled: false, type: T.PCFShadowMap}, _currentSourceMaterial: null,
    backend: {isWebGPUBackend: true, hasFeature: () => false,
      utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})}},
    hasFeature: () => false, getRenderTarget: () => null, getOutputRenderTarget: () => null,
    getMRT: () => null, currentSamples: 1, _nodes: {nodeFrame: new T.NodeFrame()}, info: {calls: 1}};
  renderer.backend.createNodeBuilder = object => new T.WGSLNodeBuilder(object, renderer);
  const programs = createThreeNodePrograms({three: T, renderer, ...options});
  const nodes = new SourceNodeManager(renderer, renderer.backend);
  const source = new SourceRenderObjects(renderer, nodes, {}, {delete() {}}, {deleteForRender() {}}, {});
  t.after(() => { programs.dispose(); source.dispose(); });
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(50, 1, .1, 30);
  camera.position.z = 4; camera.coordinateSystem = T.WebGPUCoordinateSystem;
  camera.updateProjectionMatrix(); camera.updateMatrixWorld();
  const lights = N.lights([]), renderContext = {id: 0};
  const add = material => {
    const object = new T.Mesh(new T.PlaneGeometry(), material); scene.add(object); scene.updateMatrixWorld();
    object.modelViewMatrix.multiplyMatrices(camera.matrixWorldInverse, object.matrixWorld);
    object.normalMatrix.getNormalMatrix(object.modelViewMatrix);
    return {object, material};
  };
  const context = (item, options = {}) => ({...item, scene, camera, lights: lights.getLights(), sourceLightsNode: lights,
    renderId: renderer.info.calls, frameId: renderer.info.calls, ...options});
  const compile = (item, options) => programs.compile(item.material, item.object, context(item, options));
  const capture = (item, descriptor = compile(item), options) => programs.captureDraw(descriptor, context(item, options));
  const referenceObject = (item, options = {}) => source.get(item.object, item.material, scene, camera, lights, renderContext, null, options.passId);
  const reference = (item, options) => referenceObject(item, options).getNodeBuilderState();
  const referenceDraw = (item, options = {}) => {
    const object = referenceObject(item, options);
    nodes.nodeFrame.frameId = nodes.nodeFrame.renderId = renderer.info.calls;
    if (options.fullLifecycle) nodes.updateBefore(object);
    nodes.updateForRender(object);
    const bindings = object.getBindings().flatMap(group => group.bindings);
    const textures = bindings.filter(binding => binding.isSampledTexture).map(binding => binding.textureNode.value);
    const uniforms = bindings.filter(binding => binding.isUniformsGroup).map(binding => {
      binding.update(); return new Uint8Array(binding.buffer.buffer, binding.buffer.byteOffset, binding.buffer.byteLength).slice();
    });
    if (options.fullLifecycle) nodes.updateAfter(object);
    return {textures, uniforms};
  };
  const passes = (item, run, compilation = false) => {
    const previous = [renderer._handleObjectFunction, renderer._compilationPromises, renderer._currentRenderContext];
    const results = [], queue = [], sidePass = item.material.transparent && item.material.side === T.DoubleSide && !item.material.forceSinglePass;
    const invoke = work => run({object: work.object, material: work.material},
      {passId: work.passId, sidePass, compilation, fullLifecycle: true});
    renderer._currentRenderContext = renderContext;
    renderer._compilationPromises = compilation ? queue : null;
    renderer._handleObjectFunction = compilation ? SourceRenderer.prototype._createObjectPipeline :
      (object, material, scene, camera, lightsNode, group, clippingContext, passId) => results.push(invoke({object, material, passId}));
    try {
      SourceRenderer.prototype.renderObject.call(renderer, item.object, scene, camera, item.object.geometry, item.material, null, lights);
      if (compilation) for (const work of queue) results.push(invoke(work));
      return results;
    } finally {
      [renderer._handleObjectFunction, renderer._compilationPromises, renderer._currentRenderContext] = previous;
    }
  };
  return {renderer, programs, source, scene, camera, lights, add, context, compile, capture, reference, referenceDraw, passes};
}
const values = packet => packet.uniforms.flatMap(bytes => [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);
function equalShaders(descriptor, reference) {
  assert.equal(descriptor.vertex, reference.vertexShader);
  assert.equal(descriptor.fragment, reference.fragmentShader);
}

test('extension selection and unsupported application accessors stay effect-free', t => {
  const f = fixture(t); let effects = 0;
  class Inherited extends T.MeshBasicNodeMaterial {}
  class Application extends T.MeshBasicMaterial {}
  const material = new Application();
  material.customProgramCacheKey = () => { effects++; return 'application'; };
  assert.equal(hasThreeNativeMaterialExtension(T, material, f.renderer.library), true);
  assert.equal(threeNativeMaterialExtensionKey(T, material), 'MeshBasicMaterial');
  assert.equal(threeNativeMaterialExtensionKey(T, new Inherited()), 'MeshBasicNodeMaterial');
  for (const enumerable of [false, true]) {
    class Accessor extends Application {}
    Object.defineProperty(Accessor.prototype, 'applicationValue', {enumerable, get() { effects++; return .5; }});
    const item = f.add(new Accessor());
    assert.equal(hasThreeNativeMaterialExtension(T, item.material, f.renderer.library), true);
    assert.throws(() => f.compile(item), {code: 'THREE_NODE_PROGRAM_HOOK'});
  }
  class TypeAccessor extends T.MeshBasicNodeMaterial {
    static get type() { effects++; return 'Application'; }
  }
  const opaque = Object.create(TypeAccessor.prototype);
  assert.equal(threeNativeMaterialExtensionKey(T, opaque), undefined);
  assert.equal(effects, 0);
  const opaqueItem = f.add(new TypeAccessor()), constructorEffects = effects;
  assert.throws(() => f.compile(opaqueItem), {code: 'THREE_NODE_PROGRAM_HOOK'});
  assert.equal(effects, constructorEffects);
  assert.equal(f.programs.diagnostics.programs, 0);
});

test('inherited material setup and observer use source WGSL and original frame identities', t => {
  const f = fixture(t), setups = [], observers = [], updates = [];
  class ParentMaterial extends T.MeshBasicNodeMaterial {
    constructor() {
      super();
      this.applicationTint = N.uniform(.375).setName('applicationTint').onObjectUpdate(frame => {
        updates.push([frame.object, frame.material]); return frame.material.applicationTint.value;
      });
    }
    setupOutput(builder, output) {
      setups.push([this, builder.material, builder.context.material]);
      return super.setupOutput(builder, N.vec4(output.rgb.mul(this.applicationTint), output.a));
    }
    setupObserver(builder) {
      const observer = super.setupObserver(builder);
      observer.needsRefresh = (renderObject, frame) => {
        observers.push([renderObject.material, frame.material]); return T.RenderObjectRefreshType.FULL;
      };
      return observer;
    }
  }
  class ApplicationMaterial extends ParentMaterial {}
  const item = f.add(new ApplicationMaterial()), reference = f.reference(item), descriptor = f.compile(item);
  equalShaders(descriptor, reference);
  assert.equal(setups.length, 2);
  for (const identities of setups) assert.deepEqual(identities, [item.material, item.material, item.material]);
  const first = f.capture(item, descriptor), saved = first.uniforms.map(bytes => bytes.slice());
  assert.ok(values(first).includes(.375));
  item.material.applicationTint.value = .875; f.renderer.info.calls++;
  assert.equal(f.compile(item), descriptor); assert.equal(f.reference(item), reference);
  assert.ok(values(f.capture(item, descriptor)).includes(.875));
  assert.deepEqual(first.uniforms, saved);
  for (const identities of updates) assert.deepEqual(identities, [item.object, item.material]);
  const context = {...f.context(item), bundleToken: {}, bundle: new T.BundleGroup()};
  assert.equal(f.programs.refresh(descriptor, context), T.RenderObjectRefreshType.FULL);
  assert.deepEqual(observers, [[item.material, item.material]]);
  assert.equal(setups.length, 2, 'warm uniforms and observer refresh never replay setup');
});

for (const replacement of ['custom', 'other-family', 'default'])
test(`registered material ${replacement} replacement retains source shared builders until the last owner retires`, t => {
  const f = fixture(t), trace = [];
  class Initial extends T.MeshBasicNodeMaterial {
    constructor() { super(); trace.push('initial:construct'); }
    setupOutput(builder, output) {
      trace.push(['initial:setup', this === builder.material, builder.material === builder.context.material]);
      return super.setupOutput(builder, output.mul(N.vec4(.5, .75, .25, 1)));
    }
  }
  const Base = replacement === 'other-family' ? T.MeshStandardNodeMaterial : T.MeshBasicNodeMaterial;
  class Replacement extends Base {
    constructor() { super(); trace.push('replacement:construct'); }
    setupOutput(builder, output) {
      trace.push(['replacement:setup', this === builder.material, builder.material === builder.context.material]);
      return super.setupOutput(builder, output.mul(N.vec4(.25, .5, .75, 1)));
    }
  }
  f.renderer.library.materialNodes.set('MeshBasicMaterial', Initial);
  const a = f.add(new T.MeshBasicMaterial()), sourceA = f.reference(a), sourceTrace = trace.splice(0), descriptorA = f.compile(a);
  assert.deepEqual(trace.splice(0), sourceTrace); assert.deepEqual(sourceTrace, ['initial:construct', ['initial:setup', false, true]]);
  equalShaders(descriptorA, sourceA);
  f.renderer.library.materialNodes.set('MeshBasicMaterial', replacement === 'default' ? T.MeshBasicNodeMaterial : Replacement);
  f.renderer.info.calls++;
  assert.equal(f.reference(a), sourceA); assert.equal(f.compile(a), descriptorA);
  a.material.needsUpdate = true; f.renderer.info.calls++;
  assert.equal(f.reference(a), sourceA); assert.equal(f.compile(a), descriptorA);
  const b = f.add(new T.MeshBasicMaterial()), sourceB = f.reference(b), descriptorB = f.compile(b);
  assert.equal(sourceB, sourceA); equalShaders(descriptorB, sourceB); f.capture(b, descriptorB);
  assert.deepEqual(trace, [], 'new equivalent owners reuse source construction even after registration changes');
  a.material.dispose(); f.renderer.info.calls++;
  assert.equal(f.compile(b), descriptorB);
  b.material.dispose(); f.renderer.info.calls++;
  const c = f.add(new T.MeshBasicMaterial()), sourceC = f.reference(c), replacementTrace = trace.splice(0), descriptorC = f.compile(c);
  assert.notEqual(sourceC, sourceA); assert.notEqual(descriptorC.fragment, descriptorA.fragment);
  assert.deepEqual(trace.splice(0), replacementTrace); equalShaders(descriptorC, sourceC); f.capture(c, descriptorC);
  assert.equal(f.programs.diagnostics.programs, 1);
});

test('custom material keys retain source callback counts through invalidation and disposal', t => {
  const f = fixture(t), trace = [];
  class Application extends T.MeshBasicNodeMaterial {
    customProgramCacheKey() { trace.push('key'); return `${super.customProgramCacheKey()}:${this.applicationVariant ?? 0}`; }
    setupOutput(builder, output) { trace.push('setup'); return super.setupOutput(builder, output); }
  }
  const item = f.add(new Application());
  const compare = () => {
    const source = f.reference(item), expected = trace.splice(0), descriptor = f.compile(item);
    assert.deepEqual(trace.splice(0), expected); equalShaders(descriptor, source); return descriptor;
  };
  const first = compare(); assert.equal(compare(), first);
  item.material.needsUpdate = true; f.renderer.info.calls++;
  assert.equal(compare(), first);
  item.material.applicationVariant = 1; item.material.needsUpdate = true; f.renderer.info.calls++;
  const changed = compare(); assert.notEqual(changed, first);
  item.material.dispose(); f.renderer.info.calls++;
  assert.notEqual(compare(), changed);
});

test('generated extension resources and node budgets refuse before publishing a graphics program', t => {
  const f = fixture(t); let setups = 0, updates = 0;
  class WritesStorage extends T.MeshBasicNodeMaterial {
    setupOutput(builder, output) {
      setups++;
      const storage = N.storage(new T.StorageBufferAttribute(1, 4), 'vec4', 1);
      return super.setupOutput(builder, N.Fn(() => { storage.element(0).assign(output); return output; })());
    }
  }
  const item = f.add(new WritesStorage());
  item.material.opacityNode = N.uniform(1).onObjectUpdate(() => { updates++; return 1; });
  assert.throws(() => f.compile(item), {code: 'THREE_NODE_PROGRAM_STORAGE'});
  assert.equal(setups, 1); assert.equal(updates, 0); assert.equal(f.programs.diagnostics.programs, 0);
  const bounded = fixture(t, {maxNodes: 1});
  class Application extends T.MeshBasicNodeMaterial {}
  assert.throws(() => bounded.compile(bounded.add(new Application())), {code: 'THREE_NODE_PROGRAM_LIMIT'});
  assert.equal(bounded.programs.diagnostics.programs, 0);
});

function mapTexture(value) {
  const texture = new T.DataTexture(new Uint8Array([value, 128, 255, 255]), 1, 1);
  texture.needsUpdate = true;
  return texture;
}
function environmentTexture(value) {
  const texture = new T.DataTexture(new Uint16Array(336 * 64 * 4).fill(value), 336, 64, T.RGBAFormat, T.HalfFloatType);
  texture.mapping = T.CubeUVReflectionMapping; texture.minFilter = texture.magFilter = T.LinearFilter;
  texture.colorSpace = T.LinearSRGBColorSpace; texture.needsUpdate = true;
  return texture;
}
const sampled = packet => packet.textures.filter(binding => binding.isSampledTexture).map(binding => binding.texture);

for (const physical of [false, true]) for (const registered of [false, true])
test(`${registered ? 'registered' : 'inherited'} ${physical ? 'Physical' : 'Standard'} extensions share source lit map graphs and capture live inputs`, t => {
  const f = fixture(t), trace = [];
  const Native = physical ? T.MeshPhysicalNodeMaterial : T.MeshStandardNodeMaterial;
  const Ordinary = physical ? T.MeshPhysicalMaterial : T.MeshStandardMaterial;
  class Extension extends Native {
    constructor() { super(); this.applicationGain = .375; }
    setupOutput(builder, output) {
      trace.push('setup');
      return super.setupOutput(builder, N.vec4(output.rgb.mul(N.materialReference('applicationGain', 'float')), output.a));
    }
  }
  class Application extends Ordinary {
    constructor() { super(); this.type = 'ApplicationPBRMaterial'; this.applicationGain = .375; }
  }
  if (registered) f.renderer.library.addMaterial(Extension, 'ApplicationPBRMaterial');
  const create = (gain, pixel) => {
    const material = new (registered ? Application : Extension)();
    material.applicationGain = gain; material.roughness = .5; material.metalness = .25;
    material.map = mapTexture(pixel); material.normalMap = mapTexture(128); material.roughnessMap = mapTexture(192);
    if (physical) { material.clearcoat = .5; material.clearcoatMap = mapTexture(128); }
    return f.add(material);
  };
  const ambient = new T.AmbientLight(0xffffff, .125), directional = new T.DirectionalLight(0xffffff, 2);
  directional.position.set(1, 2, 3); f.scene.add(ambient, directional, directional.target); f.scene.updateMatrixWorld();
  f.lights.setLights([directional, ambient]);
  const a = create(.375, 64), b = create(.875, 192);
  const sourceA = f.reference(a), sourceTrace = trace.splice(0), descriptorA = f.compile(a);
  assert.deepEqual(trace.splice(0), sourceTrace); equalShaders(descriptorA, sourceA);
  const sourceB = f.reference(b), descriptorB = f.compile(b);
  assert.equal(sourceB, sourceA); equalShaders(descriptorB, sourceB); assert.deepEqual(trace, []);
  const first = f.capture(a, descriptorA), saved = first.uniforms.map(bytes => bytes.slice());
  const referenceB = f.referenceDraw(b), second = f.capture(b, descriptorB);
  assert.deepEqual(sampled(second), referenceB.textures);
  assert.ok(sampled(second).includes(b.material.map)); assert.ok(!sampled(second).includes(a.material.map));
  assert.ok(values(first).includes(.375)); assert.ok(values(second).includes(.875));
  ambient.intensity = .625; directional.intensity = .75; b.material.applicationGain = .4375;
  b.material.map = mapTexture(32); f.renderer.info.calls++;
  assert.equal(f.reference(b), sourceB); assert.equal(f.compile(b), descriptorB);
  const nextSource = f.referenceDraw(b), next = f.capture(b, descriptorB);
  assert.deepEqual(sampled(next), nextSource.textures); assert.ok(values(next).includes(.4375));
  assert.deepEqual(next.uniforms, nextSource.uniforms);
  assert.deepEqual(first.uniforms, saved); assert.deepEqual(trace, []);
});

test('scene lighting keys preserve source membership, projection timing and call-level invalidation for unlit extensions', t => {
  const f = fixture(t), trace = [];
  class Extension extends T.MeshBasicNodeMaterial {
    customProgramCacheKey() { trace.push('key'); return super.customProgramCacheKey(); }
    setupOutput(builder, output) { trace.push('setup'); return super.setupOutput(builder, output); }
  }
  const item = f.add(new Extension()), ambient = new T.AmbientLight(), directional = new T.DirectionalLight();
  f.programs.inspect(item.material, item.object, f.context(item));
  f.lights.setLights([ambient]);
  const compare = () => {
    const source = f.reference(item), expected = trace.splice(0), descriptor = f.compile(item);
    assert.deepEqual(trace.splice(0), expected); equalShaders(descriptor, source); return descriptor;
  };
  const first = compare();
  f.lights.setLights([ambient, directional]);
  assert.equal(compare(), first, 'source call cache keeps a changed light list until the next renderer call');
  f.renderer.info.calls++;
  const second = compare(); assert.notEqual(second, first, 'unlit setup still observes source scene-light cache invalidation');
  f.lights.setLights([directional, ambient]); f.renderer.info.calls++;
  const reordered = compare(); assert.notEqual(reordered, second, 'projected list order participates in the exact source key');
  f.lights.setLights([]); f.renderer.info.calls++;
  assert.notEqual(compare(), reordered);
});

test('custom PBR builders retain scene and material environment precedence with source cache lifetimes', t => {
  const f = fixture(t), seen = [], trace = [];
  class Extension extends T.MeshStandardNodeMaterial {
    setupOutput(builder, output) {
      trace.push('setup'); seen.push(builder.environmentNode?.value ?? null);
      return super.setupOutput(builder, output);
    }
  }
  const sceneEnvironment = environmentTexture(0x3800), materialEnvironment = environmentTexture(0x3400);
  f.scene.environment = sceneEnvironment;
  const a = f.add(new Extension()), b = f.add(new Extension()); b.material.envMap = materialEnvironment;
  const sourceA = f.reference(a), sourceTrace = trace.splice(0), descriptorA = f.compile(a);
  assert.deepEqual(trace.splice(0), sourceTrace); equalShaders(descriptorA, sourceA);
  const sourceB = f.reference(b), sourceBTrace = trace.splice(0), descriptorB = f.compile(b);
  assert.deepEqual(trace.splice(0), sourceBTrace); equalShaders(descriptorB, sourceB);
  assert.deepEqual(seen, [sceneEnvironment, sceneEnvironment, sceneEnvironment, sceneEnvironment]);
  assert.ok(sampled(f.capture(a, descriptorA)).includes(sceneEnvironment));
  assert.ok(sampled(f.capture(b, descriptorB)).includes(materialEnvironment));
  b.material.envMap = environmentTexture(0x3000); b.material.needsUpdate = true; f.renderer.info.calls++;
  assert.equal(f.reference(b), sourceB); assert.equal(f.compile(b), descriptorB);
  assert.ok(sampled(f.capture(b, descriptorB)).includes(materialEnvironment));
  const nextSceneEnvironment = environmentTexture(0x3a00); f.scene.environment = nextSceneEnvironment; f.renderer.info.calls++;
  const replacementSource = f.reference(b), expected = trace.splice(0), replacement = f.compile(b);
  assert.deepEqual(trace.splice(0), expected); equalShaders(replacement, replacementSource);
  assert.notEqual(replacement, descriptorB);
  assert.deepEqual(seen.slice(-2), [nextSceneEnvironment, nextSceneEnvironment]);
  assert.ok(sampled(f.capture(b, replacement)).includes(b.material.envMap));
});

test('custom PBR lighting enablement retains the source environment and setup lifecycle', t => {
  const f = fixture(t), trace = [], environments = [];
  class Extension extends T.MeshPhysicalNodeMaterial {
    setupOutput(builder, output) {
      trace.push('setup'); environments.push(builder.environmentNode?.value ?? null);
      return super.setupOutput(builder, output);
    }
  }
  f.scene.environment = environmentTexture(0x3800);
  const light = new T.AmbientLight(0xffffff, .375), item = f.add(new Extension());
  f.scene.add(light); f.lights.setLights([light]);
  const compare = () => {
    const source = f.reference(item), expected = trace.splice(0), descriptor = f.compile(item);
    assert.deepEqual(trace.splice(0), expected); equalShaders(descriptor, source); return descriptor;
  };
  const lit = compare();
  f.renderer.lighting.enabled = false; f.lights.setLights([]); f.renderer.info.calls++;
  const unlit = compare(); assert.notEqual(unlit, lit);
  assert.deepEqual(environments.slice(-2), [null, null]);
  assert.equal(sampled(f.capture(item, unlit)).length, 0);
  assert.equal(compare(), unlit);
  f.renderer.lighting.enabled = true; f.lights.setLights([light]); f.renderer.info.calls++;
  const relit = compare(); assert.notEqual(relit, unlit);
  assert.deepEqual(environments.slice(-2), [f.scene.environment, f.scene.environment]);
  assert.ok(sampled(f.capture(item, relit)).includes(f.scene.environment));
});

test('an existing per-object native graph refuses registration migration before key and constructor effects', t => {
  const f = fixture(t), trace = [], material = new T.MeshBasicMaterial();
  material.colorNode = N.vec3(.25, .5, .75);
  const item = f.add(material), source = f.reference(item), descriptor = f.compile(item);
  equalShaders(descriptor, source); assert.equal(f.compile(item), descriptor);
  class Replacement extends T.MeshStandardNodeMaterial {
    constructor() { super(); trace.push('constructor'); this.fragmentNode = N.vec4(.75, .25, .5, 1); }
    setupOutput(builder, output) { trace.push('setup'); return super.setupOutput(builder, output); }
  }
  f.renderer.library.materialNodes.set('MeshBasicMaterial', Replacement);
  material.customProgramCacheKey = () => { trace.push('key'); return 'replacement'; };
  f.renderer.info.calls++;
  assert.equal(f.reference(item), source, 'source registration edits retain the existing render object and graph');
  assert.throws(() => f.compile(item), {code: 'THREE_NODE_PROGRAM_MATERIAL_CACHE'});
  assert.deepEqual(trace, []); assert.equal(f.programs.diagnostics.programs, 1);
  material.dispose(); f.renderer.info.calls++;
  const nextSource = f.reference(item), expected = trace.splice(0), next = f.compile(item);
  assert.deepEqual(trace.splice(0), expected); equalShaders(next, nextSource);
  assert.notEqual(next, descriptor); assert.notEqual(nextSource, source);
  assert.equal(f.programs.diagnostics.programs, 1);
});

for (const registered of [false, true]) for (const precompile of [false, true])
test(`${registered ? 'registered' : 'inherited'} transparent DoubleSide extensions retain source pass construction with precompile=${precompile}`, t => {
  const f = fixture(t), trace = [];
  class Extension extends T.MeshBasicNodeMaterial {
    constructor() {
      super(); trace.push(['constructor']); this.applicationVariant = 0;
      this.opacityNode = N.uniform(.375).setName('extensionAlpha').onObjectUpdate(frame => {
        trace.push(['update', frame.material.side, frame.object === item.object, frame.material === item.material]);
        return frame.material.opacity;
      });
      this.opacityNode.updateBeforeType = this.opacityNode.updateAfterType = T.NodeUpdateType.OBJECT;
      this.opacityNode.updateBefore = frame => { trace.push(['before', frame.material.side]); };
      this.opacityNode.updateAfter = frame => { trace.push(['after', frame.material.side]); };
      this.applicationGain = N.uniform(.75).setName('extensionGain').onRenderUpdate(frame => {
        trace.push(['render-update', frame.material.side]); return .75;
      });
    }
    customProgramCacheKey() { trace.push(['key', this.side]); return `${super.customProgramCacheKey()}:${this.applicationVariant}`; }
    setupOutput(builder, output) {
      trace.push(['setup', this.side, builder.material.side, builder.context.material === builder.material]);
      return super.setupOutput(builder, N.vec4(output.rgb.mul(this.applicationGain), output.a));
    }
  }
  class Application extends T.MeshBasicMaterial {
    constructor() { super(); this.type = 'TransparentApplication'; this.applicationVariant = 0; }
    customProgramCacheKey() { trace.push(['key', this.side]); return `${super.customProgramCacheKey()}:${this.applicationVariant}`; }
  }
  if (registered) f.renderer.library.addMaterial(Extension, 'TransparentApplication');
  const material = new (registered ? Application : Extension)();
  material.transparent = true; material.side = T.DoubleSide; material.opacity = .375;
  material.color.setRGB(.25, .5, .75);
  const item = f.add(material); trace.length = 0;
  const phase = compilation => {
    const reference = f.passes(item, (selected, options) => ({state: f.reference(selected, options), packet: f.referenceDraw(selected, options)}), compilation);
    const expected = trace.splice(0);
    const candidate = f.passes(item, (selected, options) => {
      const descriptor = f.compile(selected, options), context = f.context(selected, options);
      f.programs.before(descriptor, context);
      const packet = f.capture(selected, descriptor, {...options, beforeDone: true});
      f.programs.after(descriptor, context);
      return {descriptor, packet};
    }, compilation);
    assert.deepEqual(trace.splice(0), expected, 'cache keys, constructors, setup and NodeFrame effects match source order');
    assert.equal(candidate.length, reference.length);
    candidate.forEach(({descriptor, packet}, index) => {
      equalShaders(descriptor, reference[index].state);
      assert.deepEqual(packet.uniforms, reference[index].packet.uniforms);
    });
    assert.equal(material.side, T.DoubleSide);
    return {reference, candidate, trace: expected};
  };
  const first = phase(precompile), saved = first.candidate.map(({packet}) => packet.uniforms.map(bytes => bytes.slice()));
  assert.equal(first.trace.filter(event => event[0] === 'setup').length, precompile ? 1 : 2);
  assert.deepEqual(first.trace.filter(event => event[0] === 'key').map(event => event[1]),
    precompile ? [T.DoubleSide, T.DoubleSide] : [T.BackSide, T.FrontSide]);
  assert.equal(first.reference[0].state === first.reference[1].state, precompile);
  assert.notEqual(first.candidate[0].descriptor, first.candidate[1].descriptor, 'each pass retains independent binding ownership');
  material.opacity = .8125; material.color.setRGB(.75, .25, .5); f.renderer.info.calls++;
  const warm = phase(false);
  assert.equal(warm.trace.some(event => event[0] === 'setup' || event[0] === 'key'), false);
  for (let index = 0; index < 2; index++) {
    assert.equal(warm.reference[index].state, first.reference[index].state);
    assert.equal(warm.candidate[index].descriptor, first.candidate[index].descriptor);
    assert.ok(values(warm.candidate[index].packet).includes(.8125));
    assert.deepEqual(first.candidate[index].packet.uniforms, saved[index]);
  }
  f.renderer.info.calls++;
  const compiledAfterDraw = phase(true);
  assert.equal(compiledAfterDraw.trace.some(event => event[0] === 'setup' || event[0] === 'key'), false);
  material.needsUpdate = true; f.renderer.info.calls++;
  const invalidated = phase(false);
  assert.equal(invalidated.trace.filter(event => event[0] === 'setup').length, precompile ? 2 : 0);
  material.applicationVariant = 1; material.needsUpdate = true; f.renderer.info.calls++;
  const rebuilt = phase(true);
  assert.equal(rebuilt.trace.filter(event => event[0] === 'setup').length, 1);
  assert.equal(rebuilt.reference[0].state, rebuilt.reference[1].state);
  material.forceSinglePass = true; f.renderer.info.calls++;
  const single = phase(false);
  assert.equal(single.candidate.length, 1);
  assert.equal(single.candidate[0].descriptor, rebuilt.candidate[1].descriptor);
  assert.equal(single.trace.some(event => event[0] === 'setup' || event[0] === 'key'), false);
  material.needsUpdate = true; f.renderer.info.calls++;
  assert.equal(phase(false).trace.filter(event => event[0] === 'setup').length, 1);
  material.forceSinglePass = false; material.dispose(); f.renderer.info.calls++;
  assert.equal(f.programs.diagnostics.programs, 0);
  assert.equal(phase(false).trace.filter(event => event[0] === 'setup').length, 2);
});
