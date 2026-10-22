import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms, hasThreeNodeProgram} from './three_node_program.mjs';
const N = THREE.TSL;
function rig(t, options = {}) {
  const frame = new THREE.NodeFrame();
  const renderer = {library: new THREE.StandardNodeLibrary(), contextNode: N.context(),
    debug: {diagnostics: {keywords: false}}, depth: true, lighting: {enabled: true},
    backend: {hasFeature: () => false}, hasFeature: () => false, getRenderTarget: () => null,
    getMRT: () => null, currentSamples: 1, _nodes: {nodeFrame: frame}, info: {calls: 1}};
  const owner = createThreeNodePrograms({three: THREE, renderer, ...options}); t.after(() => owner.dispose());
  const material = new THREE.MeshBasicNodeMaterial();
  const tint = N.uniform(new THREE.Color(.25, .5, .75)).setName('testTint');
  const shift = N.uniform(.125).setName('testShift');
  material.colorNode = tint; material.positionNode = N.positionLocal.add(N.vec3(shift, 0, 0));
  const object = new THREE.Mesh(new THREE.PlaneGeometry(), material), scene = new THREE.Scene(); scene.add(object);
  const camera = new THREE.PerspectiveCamera(60, 1, .1, 100);
  camera.coordinateSystem = THREE.WebGPUCoordinateSystem; camera.position.z = 4; camera.updateProjectionMatrix();
  scene.updateMatrixWorld(); camera.updateMatrixWorld();
  const context = {object, material, scene, camera, renderId: 1, frameId: 1, time: .2, deltaTime: .01};
  return {owner, renderer, frame, material, object, scene, camera, tint, shift, context,
    compile: () => owner.compile(material, object, context)};
}
const floats = packets => packets.flatMap(bytes => Array.from(new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)));
test('real pinned node graphs preserve native WGSL groups, attributes and linear output', t => {
  const f = rig(t), d = f.compile();
  assert.equal(d.nodeProgram, true); assert.equal('reflection' in d, false);
  assert.match(d.vertex, /@vertex\s+fn main/); assert.match(d.fragment, /@fragment\s+fn main/);
  assert.match(d.fragment, /object\.testTint/); assert.doesNotMatch(d.fragment, /sRGBTransferOETF|LinearTosRGB/);
  assert.deepEqual(d.groups.map(g => g.name), ['render', 'object']);
  assert.deepEqual(d.attributes, [{name: 'position', location: 0, locations: 1, components: 3, scalar: 'float'}]);
  const captured = f.owner.capture(d, f.context);
  for (const v of [.25, .5, .75, .125]) assert.ok(floats(captured).includes(v));
  assert.equal(hasThreeNodeProgram(f.material), true);
  assert.equal(hasThreeNodeProgram(new THREE.MeshBasicNodeMaterial()), false);
});
test('uniform value edits reuse the program while old packets remain immutable', t => {
  const f = rig(t), d = f.compile(), first = f.owner.capture(d, f.context), saved = first.map(b => b.slice());
  f.tint.value.setRGB(.875, .625, .375); f.shift.value = .75;
  assert.equal(f.compile(), d);
  const second = f.owner.capture(d, {...f.context, renderId: 2});
  assert.deepEqual(first, saved); assert.ok(floats(second).includes(.875)); assert.notDeepEqual(second, first);
  f.material.colorNode = N.vec3(.3, .2, .1);
  assert.notEqual(f.compile(), d); assert.equal(f.owner.diagnostics.programs, 1);
});
test('matrix, integer and boolean uniforms retain padded native values', t => {
  const f = rig(t), transform = N.uniform(new THREE.Matrix3().set(2, 0, 0, 0, 3, 0, 0, 0, 4)).setName('testMatrix');
  const choice = N.uniform(true, 'bool').setName('testChoice'), value = N.uniform(7, 'int').setName('testInteger');
  f.material.positionNode = transform.mul(N.positionLocal);
  f.material.fragmentNode = N.vec4(N.select(choice, value.toFloat(), N.float(2)), 0, 0, 1);
  const d = f.compile(), packets = f.owner.capture(d, f.context);
  assert.match(d.vertex, /mat3x3<f32>/); assert.match(d.fragment, /i32/);
  assert.ok(floats(packets).includes(3)); assert.ok(floats(packets).includes(4));
  assert.ok(packets.some(b => Array.from(new Int32Array(b.buffer)).includes(7)));
});
test('custom float attributes keep builder-assigned locations', t => {
  const f = rig(t);
  f.object.geometry.setAttribute('weight', new THREE.Float32BufferAttribute([.1, .2, .3, .4], 1));
  f.material.fragmentNode = N.vec4(N.attribute('weight', 'float'), N.uv(), 1);
  const d = f.compile();
  assert.ok(d.attributes.some(a => a.name === 'weight' && a.components === 1));
  assert.ok(d.attributes.some(a => a.name === 'uv' && a.components === 2));
});
test('base unlit NodeMaterial supports clip-space vertex and fragment graphs with no uniform groups', t => {
  const f = rig(t), material = new THREE.NodeMaterial();
  material.vertexNode = N.vec4(N.attribute('position', 'vec3'), 1);
  material.fragmentNode = N.vec4(.25, .5, .75, 1);
  const d = f.owner.compile(material, f.object, f.context);
  assert.equal(d.groups.length, 0); assert.equal(d.attributes[0].name, 'position');
  assert.deepEqual(f.owner.capture(d, {...f.context, material}), []);
});
test('source NodeFrame preserves FRAME, RENDER and OBJECT updates and all context identities', t => {
  const f = rig(t), calls = {frame: 0, render: 0, object: 0}, seen = [];
  const nodes = Object.keys(calls).map(kind => {
    const node = N.uniform(0); node.updateType = kind;
    node.update = current => { calls[kind]++; seen.push({renderer: current.renderer, scene: current.scene,
      camera: current.camera, object: current.object, material: current.material, compute: current.compute}); node.value = calls[kind]; };
    return node;
  });
  f.material.colorNode = N.vec3(...nodes); const d = f.compile();
  f.owner.capture(d, f.context); f.owner.capture(d, f.context);
  assert.deepEqual(calls, {frame: 1, render: 1, object: 2});
  f.owner.capture(d, {...f.context, renderId: 2});
  assert.deepEqual(calls, {frame: 1, render: 2, object: 3});
  f.owner.capture(d, {...f.context, renderId: 3, frameId: 2});
  assert.deepEqual(calls, {frame: 2, render: 3, object: 4});
  for (const context of seen) assert.deepEqual(context, {renderer: f.renderer, scene: f.scene,
    camera: f.camera, object: f.object, material: f.material, compute: null});
});
test('before/current/after false returns retain the pinned retry-marker behavior', t => {
  const f = rig(t), node = N.uniform(0), calls = {before: 0, current: 0, after: 0};
  node.updateBeforeType = 'render'; node.updateType = 'render'; node.updateAfterType = 'render';
  node.updateBefore = () => ++calls.before === 1 ? false : undefined;
  node.update = () => { node.value = ++calls.current; return calls.current === 1 ? false : undefined; };
  node.updateAfter = () => ++calls.after === 1 ? false : undefined;
  f.material.colorNode = N.vec3(node); const d = f.compile();
  for (let i = 0; i < 3; i++) { f.owner.capture(d, f.context); f.owner.after(d, f.context); }
  assert.deepEqual(calls, {before: 2, current: 2, after: 2});
});
test('shared update references are updated once per render across distinct nodes', t => {
  const f = rig(t), reference = {}, a = N.uniform(.1), b = N.uniform(.2); let calls = 0;
  for (const node of [a, b]) { node.updateType = 'render'; node.updateReference = () => reference; node.update = () => { calls++; }; }
  f.material.colorNode = N.vec3(a, b, 0); const d = f.compile(); f.owner.capture(d, f.context);
  assert.equal(calls, 1);
});
test('same material on two objects captures different object-update values', t => {
  const f = rig(t), other = new THREE.Mesh(f.object.geometry, f.material); other.position.x = 2; other.updateMatrixWorld();
  f.material.colorNode = N.uniform(new THREE.Color()).onObjectUpdate(({object}, uniform) => uniform.value.setRGB(object === f.object ? .25 : .75, .5, 0));
  const d1 = f.compile(), d2 = f.owner.compile(f.material, other, {...f.context, object: other});
  const first = f.owner.capture(d1, f.context), second = f.owner.capture(d2, {...f.context, object: other});
  assert.ok(floats(first).includes(.25)); assert.ok(floats(second).includes(.75));
  assert.notDeepEqual(first, second);
});
test('nested captures and thrown callbacks restore the exact outer NodeFrame context', t => {
  const f = rig(t), innerMaterial = new THREE.MeshBasicNodeMaterial(), innerObject = new THREE.Mesh(f.object.geometry, innerMaterial);
  innerMaterial.colorNode = N.vec3(.2); const inner = f.owner.compile(innerMaterial, innerObject, f.context);
  const initial = Object.fromEntries(['renderer', 'scene', 'camera', 'object', 'material', 'compute', 'frameId', 'renderId', 'time', 'deltaTime'].map(k => [k, f.frame[k]]));
  const failure = Error('callback failure'); let nested = false;
  f.material.colorNode = N.uniform(0).onObjectUpdate(current => {
    f.owner.capture(inner, {...f.context, object: innerObject, material: innerMaterial, renderId: 2});
    assert.equal(current.object, f.object); assert.equal(current.material, f.material); assert.equal(current.renderId, 1);
    nested = true; throw failure;
  });
  assert.throws(() => f.owner.capture(f.compile(), f.context), e => e === failure); assert.equal(nested, true);
  assert.deepEqual(Object.fromEntries(Object.keys(initial).map(k => [k, f.frame[k]])), initial);
});
for (const [name, change, code] of [
  ['texture graph', f => { f.material.colorNode = N.texture(new THREE.DataTexture(new Uint8Array(4), 1, 1)); }, 'TEXTURE'],
  ['storage graph', f => { f.material.colorNode = N.storage(new THREE.StorageBufferAttribute(4, 4), 'vec4', 4).element(0).xyz; }, 'STORAGE'],
  ['material texture', f => { f.material.map = new THREE.Texture(); }, 'TEXTURE'],
  ['MRT graph', f => { f.material.fragmentNode = N.mrt({output: N.vec4(1)}); }, 'MRT'],
  ['lighting slot', f => { f.material.lightsNode = N.vec3(1); }, 'GRAPH'],
  ['transparent double side', f => { f.material.side = THREE.DoubleSide; f.material.transparent = true; }, 'RASTER'],
  ['custom renderer context', f => { f.renderer.contextNode = N.context({getOutput: output => output}); }, 'GRAPH'],
  ['instancing', f => { f.object.isInstancedMesh = true; }, 'OBJECT'],
  ['fog', f => { f.scene.fog = new THREE.Fog(0, 1, 10); }, 'EFFECT'],
]) test(`${name} is explicitly refused`, t => {
  const f = rig(t); change(f); assert.throws(f.compile, {code: `THREE_NODE_PROGRAM_${code}`});
});
test('source missing attributes retain the builder constant fallback without a fake vertex stream', t => {
  const f = rig(t); f.material.positionNode = N.attribute('absent', 'vec3');
  const d = f.compile(); assert.equal(d.attributes.some(a => a.name === 'absent'), false);
  assert.match(d.vertex, /positionLocal = vec3<f32>\( 0\.0, 0\.0, 0\.0 \)/);
});
test('program budget, material disposal and owner identity are enforced', t => {
  const f = rig(t, {maxPrograms: 1}), d = f.compile(), other = new THREE.Mesh(f.object.geometry, f.material);
  assert.throws(() => f.owner.compile(f.material, other, f.context), {code: 'THREE_NODE_PROGRAM_LIMIT'});
  f.material.dispose(); assert.equal(f.owner.diagnostics.programs, 0);
  assert.ok(f.owner.compile(f.material, other, f.context));
  const foreign = rig(t); assert.throws(() => foreign.owner.capture(d, f.context), {code: 'THREE_NODE_PROGRAM_SOURCE'});
  f.owner.dispose(); assert.throws(f.compile, {code: 'THREE_NODE_PROGRAM_DISPOSED'});
});
