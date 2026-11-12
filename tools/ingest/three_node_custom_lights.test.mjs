/** Registered analytic lights use the pinned builder and NodeFrame lifecycle.
 * These comparisons check source WGSL, callbacks and captured bytes, not pixels. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';

const N = T.TSL;
function fixture(t) {
  const frame = new T.NodeFrame();
  const renderer = {library: new T.StandardNodeLibrary(), contextNode: N.context(),
    debug: {diagnostics: {keywords: false}}, depth: true, lighting: {enabled: true}, shadowMap: {enabled: false},
    backend: {hasFeature: () => false, utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})},
      capabilities: {getUniformBufferLimit: () => 65536}},
    hasFeature: () => false, getRenderTarget: () => null, getMRT: () => null, currentSamples: 1,
    _nodes: {nodeFrame: frame}, info: {calls: 1}};
  const programs = createThreeNodePrograms({three: T, renderer});
  const material = new T.MeshLambertNodeMaterial(), geometry = new T.PlaneGeometry();
  const object = new T.Mesh(geometry, material), scene = new T.Scene(); object.name = 'first'; scene.add(object);
  const camera = new T.PerspectiveCamera(50, 1, .1, 100); camera.position.z = 4;
  camera.coordinateSystem = T.WebGPUCoordinateSystem; camera.updateProjectionMatrix(); camera.updateMatrixWorld();
  scene.updateMatrixWorld();
  const context = {object, material, scene, camera, frameId: 1, renderId: 1, time: .25, deltaTime: .02};
  t.after(() => { programs.dispose(); geometry.dispose(); material.dispose(); });
  return {programs, renderer, frame, material, object, scene, camera, context};
}
function reference(f, lights, object = f.object) {
  const builder = new T.WGSLNodeBuilder(object, f.renderer);
  builder.scene = f.scene; builder.camera = f.camera; builder.lightsNode = N.lights(lights); builder.build();
  return builder;
}
const floats = packets => packets.flatMap(bytes =>
  [...new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)]);
const hasVector = (packets, expected) => {
  const values = floats(packets);
  return values.some((_value, index) => expected.every((value, offset) => values[index + offset] === value));
};

test('registered inherited ambient nodes retain source WGSL and render-frequency uniforms', t => {
  const f = fixture(t); let constructed = 0;
  class CustomLight extends T.AmbientLight {}
  class CustomNode extends T.AmbientLightNode {
    constructor(light) { super(light); constructed++; this.updateType = 'render'; this.colorNode.setName('registeredLightColor'); }
    update(frame) { super.update(frame); this.color.multiplyScalar(frame.camera.userData.gain); }
  }
  f.renderer.library.addLight(CustomNode, CustomLight);
  const light = new CustomLight(); light.color.setRGB(.25, .5, .75); light.intensity = 2;
  f.scene.add(light); f.scene.updateMatrixWorld(); f.camera.userData.gain = .25;
  const source = reference(f, [light]), context = {...f.context, lights: [light]};
  const descriptor = f.programs.compile(f.material, f.object, context);
  assert.equal(descriptor.vertex, source.vertexShader);
  assert.equal(descriptor.fragment, source.fragmentShader);
  assert.match(descriptor.fragment, /registeredLightColor/);
  const first = f.programs.capture(descriptor, context), saved = first.map(bytes => bytes.slice());
  assert.ok(hasVector(first, [.125, .25, .375]));
  f.camera.userData.gain = .5;
  const second = f.programs.capture(descriptor, {...context, renderId: 2});
  assert.ok(hasVector(second, [.25, .5, .75]));
  assert.deepEqual(first, saved, 'later light updates cannot rewrite accepted uniform packets');
  assert.equal(f.programs.compile(f.material, f.object, context), descriptor);
  assert.equal(constructed, 1, 'both builders use the source light-identity cache');
});

for (const frequency of ['frame', 'render', 'object'])
test(`registered analytic before/current/after updates retain source ${frequency} scheduling`, t => {
  const f = fixture(t), trace = []; let constructed = 0;
  class BaseLight extends T.Light {}
  class CustomLight extends BaseLight {}
  const record = (phase, frame) => trace.push({phase, frameId: frame.frameId, renderId: frame.renderId,
    object: frame.object, material: frame.material, scene: frame.scene, camera: frame.camera, renderer: frame.renderer});
  class BaseNode extends T.AnalyticLightNode {
    direction() { return N.vec3(0, 1, 0); }
    setupDirect() { return {lightDirection: this.direction(), lightColor: this.colorNode}; }
    updateBefore(frame) { record('before', frame); }
    update(frame) { record('update', frame); super.update(frame); }
    updateAfter(frame) { record('after', frame); }
  }
  class CustomNode extends BaseNode {
    constructor(light) {
      super(light); constructed++;
      this.updateBeforeType = this.updateType = this.updateAfterType = frequency;
    }
  }
  f.renderer.library.addLight(CustomNode, CustomLight);
  const light = new CustomLight(); f.scene.add(light);
  const other = new T.Mesh(f.object.geometry, f.material); other.name = 'second'; f.scene.add(other); f.scene.updateMatrixWorld();
  const source = reference(f, [light]), sourceFrame = new T.NodeFrame();
  const firstContext = {...f.context, lights: [light]}, secondContext = {...firstContext, object: other};
  const first = f.programs.compile(f.material, f.object, firstContext);
  const second = f.programs.compile(f.material, other, secondContext);
  const calls = [[first, firstContext], [second, secondContext], [first, {...firstContext, renderId: 2}],
    [first, {...firstContext, frameId: 2, renderId: 3}]];
  for (const [, context] of calls) {
    Object.assign(sourceFrame, context, {renderer: f.renderer});
    for (const node of source.updateBeforeNodes) sourceFrame.updateBeforeNode(node);
    for (const node of source.updateNodes) sourceFrame.updateNode(node);
    for (const node of source.updateAfterNodes) sourceFrame.updateAfterNode(node);
  }
  const expected = trace.splice(0);
  for (const [descriptor, context] of calls) {
    f.programs.before(descriptor, context);
    f.programs.captureDraw(descriptor, {...context, beforeDone: true});
    f.programs.after(descriptor, context);
  }
  assert.deepEqual(trace, expected, 'the retained NodeFrame owns update frequency, ordering and source identities');
  assert.equal(trace.length, {frame: 6, render: 9, object: 12}[frequency]);
  assert.ok(trace.every(event => event.material === f.material && event.scene === f.scene &&
    event.camera === f.camera && event.renderer === f.renderer));
  assert.equal(constructed, 1, 'separate objects retain the original analytic light node');
});

test('custom analytic setup still rejects graphics storage writes before update callbacks', t => {
  const f = fixture(t); let updates = 0;
  class CustomLight extends T.Light {}
  class CustomNode extends T.AnalyticLightNode {
    constructor(light) { super(light); this.updateBeforeType = this.updateType = this.updateAfterType = 'object'; }
    setupDirect() {
      const storage = N.storage(new T.StorageBufferAttribute(1, 4), 'vec4', 1);
      return {lightDirection: N.vec3(0, 1, 0), lightColor: N.Fn(() => {
        storage.element(0).assign(N.vec4(1)); return this.colorNode;
      })()};
    }
    updateBefore() { updates++; }
    update() { updates++; }
    updateAfter() { updates++; }
  }
  f.renderer.library.addLight(CustomNode, CustomLight);
  const light = new CustomLight(); f.scene.add(light);
  assert.throws(() => f.programs.compile(f.material, f.object, {...f.context, lights: [light]}),
    {code: 'THREE_NODE_PROGRAM_STORAGE'});
  assert.equal(updates, 0);
  assert.equal(f.programs.diagnostics.programs, 0);
});

test('registry replacements retain cached source light nodes without replaying warm setup', t => {
  const f = fixture(t), constructed = [0, 0], setups = [0, 0];
  class OriginalNode extends T.AmbientLightNode {
    constructor(light) { super(light); constructed[0]++; this.colorNode.setName('registeredAmbient'); }
    setup(builder) { setups[0]++; super.setup(builder); }
  }
  class ReplacementNode extends T.AmbientLightNode {
    constructor(light) { super(light); constructed[1]++; this.colorNode.setName('registeredAmbient'); }
    setup(builder) { setups[1]++; builder.context.irradiance.addAssign(this.colorNode.mul(.5)); }
  }
  f.renderer.library.lightNodes.set(T.AmbientLight, OriginalNode);
  const light = new T.AmbientLight(); f.scene.add(light);
  const source = reference(f, [light]), context = {...f.context, lights: [light]};
  const first = f.programs.compile(f.material, f.object, context);
  assert.equal(first.fragment, source.fragmentShader);
  f.programs.capture(first, context);
  const setupCount = setups.slice();
  f.renderer.library.lightNodes.set(T.AmbientLight, ReplacementNode);
  assert.equal(f.programs.compile(f.material, f.object, context), first,
    'the source registry is not a warm program invalidation key');
  assert.doesNotThrow(() => f.programs.capture(first, {...context, renderId: 2}));
  assert.deepEqual(setups, setupCount, 'a mapping replacement cannot replay setup on the cached original node');
  assert.deepEqual(constructed, [1, 0]);
  const rebuiltSource = reference(f, [light]);
  assert.equal(rebuiltSource.fragmentShader, first.fragment, 'a fresh source builder still uses the old light node');
  assert.deepEqual(constructed, [1, 0]);
  const another = fixture(t); another.renderer.library.lightNodes.set(T.AmbientLight, ReplacementNode);
  const acrossOwners = another.programs.compile(another.material, another.object, {...another.context, lights: [light]});
  assert.equal(acrossOwners.fragment, first.fragment, 'the source light cache is shared across renderer owners');
  assert.deepEqual(constructed, [1, 0]);
  const fresh = new T.AmbientLight(), freshContext = {...context, lights: [fresh]};
  f.scene.add(fresh);
  const freshSource = reference(f, [fresh]);
  const next = f.programs.compile(f.material, f.object, freshContext);
  assert.equal(next.fragment, freshSource.fragmentShader);
  assert.notEqual(next.fragment, first.fragment);
  assert.deepEqual(constructed, [1, 1], 'a new light identity uses the replacement mapping');
});
