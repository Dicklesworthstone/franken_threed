import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';
import {captureThreeNodeScene} from './three_node_scene_capture.mjs';
import {withThreeNativeCapture, assertThreeNativeTargetWrite, assertThreeNativeComputeOutsideCapture} from './three_node_shadows.mjs';
import {createGpuCompute} from './gpu_compute.mjs';
import {captureComputeStorage} from './gpu_compute_attributes.mjs';
import {geometryDevice} from './fixtures/gpu_geometry_device.mjs';
import ClippingContext from '../../upstream/three.js/src/renderers/common/ClippingContext.js';
import {captureThreeNodeClipping} from './three_node_clipping.mjs';
const N = THREE.TSL;
function rig(t) {
  const frame = new THREE.NodeFrame(); Object.assign(frame, {frameId: 7, time: 1.25, deltaTime: .125});
  const renderer = {library: new THREE.StandardNodeLibrary(), contextNode: N.context(),
    debug: {diagnostics: {keywords: false}}, depth: true, lighting: {enabled: true}, sortObjects: true,
    backend: {hasFeature: () => false}, hasFeature: () => false, getRenderTarget: () => null,
    getMRT: () => null, currentSamples: 1, _nodes: {nodeFrame: frame}, info: {calls: 3}};
  const owner = createThreeNodePrograms({three: THREE, renderer}); t.after(() => owner.dispose());
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(60, 1, .1, 100); camera.position.z = 5;
  const material = () => { const m = new THREE.MeshBasicNodeMaterial(); m.colorNode = N.vec3(.25, .5, .75); return m; };
  const mesh = (m = material(), parent = scene) => { const o = new THREE.Mesh(new THREE.PlaneGeometry(), m); parent.add(o); return o; };
  const capture = (options = {}) => captureThreeNodeScene(THREE, scene, camera, {
    getPrograms: () => owner, renderer, raster: m => Object.freeze({side: m.side, depthWrite: m.depthWrite}), ...options});
  return {owner, frame, renderer, scene, camera, material, mesh, capture};
}
const floats = packets => packets.flatMap(bytes => Array.from(new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)));
const planeValues = context => Object.fromEntries(['unionPlanes', 'intersectionPlanes'].map(key =>
  [key, (context?.[key] ?? []).map(plane => plane.toArray())]));
test('nested native ClippingGroups match source camera projection, union/intersection and layer inheritance', t => {
  const f = rig(t), outer = new THREE.ClippingGroup(), inner = new THREE.ClippingGroup(), excluded = new THREE.ClippingGroup();
  outer.clippingPlanes = [new THREE.Plane(new THREE.Vector3(1, 0, 0), .25)];
  inner.clippingPlanes = [new THREE.Plane(new THREE.Vector3(0, 1, 0), .5), new THREE.Plane(new THREE.Vector3(0, 0, 1), .75)];
  inner.clipIntersection = true; excluded.clippingPlanes = [new THREE.Plane(new THREE.Vector3(1, 0, 0), 99)];
  excluded.layers.set(1); f.scene.add(outer); outer.add(inner); inner.add(excluded);
  outer.position.set(20, 30, 40); inner.scale.set(2, 3, 4); // groups do not transform their world-space planes
  const material = new THREE.MeshBasicMaterial(), a = f.mesh(material, outer), b = f.mesh(material, excluded);
  f.camera.position.set(2, 3, 5); f.camera.lookAt(0, 0, 0);
  const captured = f.capture(), source = new ClippingContext(); source.updateGlobal(f.scene, f.camera);
  const sourceOuter = source.getGroupContext(outer), sourceInner = sourceOuter.getGroupContext(inner);
  const binding = object => captured.bindings.find(value => value.object === object);
  assert.deepEqual(planeValues(binding(a).clippingContext), planeValues(sourceOuter));
  assert.deepEqual(planeValues(binding(b).clippingContext), planeValues(sourceInner));
  assert.ok(captured.objects.has(a) && captured.objects.has(b), 'ordinary materials enter native clipping execution');
  assert.notEqual(binding(a).nodeProgram, binding(b).nodeProgram, 'shared source materials retain distinct clipping layouts');
  assert.ok(captured.draws.every(draw => draw.binding.nodeProgram.fragment.includes('discard')));
});
test('projected clip planes precede draw callbacks and keep separate accepted render snapshots', t => {
  const f = rig(t), group = new THREE.ClippingGroup(); group.clippingPlanes = [new THREE.Plane(new THREE.Vector3(1, 0, 0), .25)];
  f.scene.add(group); const material = f.material(); let updates = 0;
  material.colorNode = N.uniform(.5).onObjectUpdate(() => {
    updates++; group.clippingPlanes[0].constant = updates + 1; return .5;
  });
  const object = f.mesh(material, group), first = f.capture(), draw = first.draws[0];
  const saved = draw.nodeUniforms.map(bytes => bytes.slice());
  assert.equal(draw.binding.clippingContext.unionPlanes[0].w, .25, 'source projection was captured before the callback');
  f.renderer.info.calls++; f.camera.position.x = 1;
  const second = f.capture();
  assert.equal(second.draws[0].binding.nodeProgram, draw.binding.nodeProgram, 'moving camera and planes retain the layout');
  assert.deepEqual(draw.nodeUniforms, saved); assert.notDeepEqual(second.draws[0].nodeUniforms, saved);
  assert.equal(second.draws[0].binding.clippingContext.unionPlanes[0].w, 3);
  group.enabled = false; f.renderer.info.calls++;
  const disabled = f.capture();
  assert.equal(disabled.draws[0].object, object); assert.deepEqual(planeValues(disabled.bindings[0].clippingContext),
    {unionPlanes: [], intersectionPlanes: []});
  assert.notEqual(disabled.bindings[0].nodeProgram, draw.binding.nodeProgram);
});
test('native shadow clipping follows the source clipShadows ancestry rule', () => {
  const scene = new THREE.Scene(), outer = new THREE.ClippingGroup(), inner = new THREE.ClippingGroup();
  const camera = new THREE.PerspectiveCamera(); camera.position.set(1, 2, 3); camera.updateMatrixWorld();
  outer.clippingPlanes = [new THREE.Plane(new THREE.Vector3(1, 0, 0), .25)];
  inner.clippingPlanes = [new THREE.Plane(new THREE.Vector3(0, 1, 0), .5)];
  inner.clipShadows = true; inner.clipIntersection = true; scene.add(outer); outer.add(inner);
  scene.overrideMaterial = {isShadowPassMaterial: true};
  const hierarchy = new Map([[scene, [outer]], [outer, [inner]], [inner, []]]), visibility = new Map([...hierarchy.keys()].map(o => [o, true]));
  const capture = () => captureThreeNodeClipping(THREE, {root: scene, camera, hierarchy, visibility, shadowPass: true,
    fail: (_, message) => assert.fail(message)});
  let contexts = capture(), source = new ClippingContext(); source.updateGlobal(scene, camera);
  assert.deepEqual(planeValues(contexts.get(inner)), planeValues(source.getGroupContext(outer).getGroupContext(inner)));
  assert.equal(contexts.get(inner).unionPlanes.length, 0);
  outer.clipShadows = true; contexts = capture(); source = new ClippingContext(); source.updateGlobal(scene, camera);
  assert.deepEqual(planeValues(contexts.get(inner)), planeValues(source.getGroupContext(outer).getGroupContext(inner)));
  assert.equal(contexts.get(inner).unionPlanes.length, 1);
});
test('ordinary scenes do not create a node compiler or change the source camera', t => {
  const f = rig(t); f.mesh(new THREE.MeshBasicMaterial());
  assert.equal(f.capture({getPrograms() { assert.fail('unexpected native compiler'); }}), null);
  assert.equal(f.camera.coordinateSystem, THREE.WebGLCoordinateSystem); assert.equal(f.frame.frameId, 7);
});
test('native shadow intent promotes default mesh materials without changing fixed-only scenes', t => {
  const fixed = rig(t); fixed.renderer.shadowMap = {enabled: true}; fixed.mesh(new THREE.MeshBasicMaterial());
  assert.equal(fixed.capture({getPrograms() { assert.fail('fixed-only shadows keep their existing owner'); }}), null);
  for (const authored of [false, true]) {
    const f = rig(t); f.renderer.shadowMap = {enabled: true};
    const ordinary = f.mesh(new THREE.MeshBasicMaterial()), native = f.mesh(authored ? f.material() : new THREE.MeshBasicNodeMaterial());
    const result = f.capture();
    assert.deepEqual([...result.objects], [ordinary, native]);
    assert.equal(result.bindings[0].material, ordinary.material); assert.equal(result.bindings.length, 2);
    assert.deepEqual(result.draws.map(draw => draw.object), [ordinary, native]);
  }
});
test('unsupported mixed shadow materials are re-admitted before updates when they become visible', t => {
  const f = rig(t); f.renderer.shadowMap = {enabled: true};
  const material = f.material(); let updates = 0;
  material.colorNode = N.uniform(.5).onObjectUpdate(() => { updates++; return .5; }); f.mesh(material);
  const hidden = f.mesh(new THREE.ShaderMaterial()); hidden.visible = false;
  const captured = f.capture(); assert.ok(captured.excludedObjects.has(hidden)); assert.equal(updates, 1);
  hidden.visible = true;
  assert.throws(() => f.capture(), {code: 'THREE_NODE_PROGRAM_MATERIAL'}); assert.equal(updates, 1);
});
test('private native backgrounds precede scene draws independently of sorting, layers and overrides', t => {
  for (const sortObjects of [false, true]) {
    const f = rig(t), seen = [], foreground = f.mesh(), fixed = f.mesh(new THREE.MeshBasicMaterial());
    foreground.renderOrder = -100; f.renderer.sortObjects = sortObjects;
    const override = f.material();
    override.colorNode = N.uniform(.75).onObjectUpdate(frame => { seen.push(frame.object); return .75; });
    const material = f.material();
    const background = new THREE.Mesh(new THREE.PlaneGeometry(), material);
    background.visible = false; background.layers.set(4); background.renderOrder = 100;
    background.position.x = 100; // projection traversal would also cull it
    material.colorNode = N.uniform(.25).onObjectUpdate(frame => {
      assert.equal(frame.scene, f.scene); assert.equal(frame.camera, f.camera);
      assert.equal(frame.object, background); assert.equal(frame.material, material);
      assert.equal(frame.object.modelViewMatrix.elements[12], 100);
      assert.equal(frame.object.modelViewMatrix.elements[14], -5);
      seen.push(frame.object); return .25;
    });
    const children = f.scene.children.slice();
    const captured = f.capture({backgroundMesh: background, overrideMaterial: override});
    assert.equal(captured.background, true);
    assert.equal(captured.bindings[0].object, background);
    assert.equal(captured.bindings[0].material, material, 'scene override cannot replace the private background material');
    assert.deepEqual(captured.draws.map(draw => draw.object), [background, foreground, fixed]);
    assert.deepEqual(captured.draws.map(draw => draw.background), [true, false, false]);
    assert.deepEqual(seen, [background, foreground, fixed]);
    assert.deepEqual(f.scene.children, children); assert.equal(background.parent, null);
    assert.ok(children.every(child => child.parent === f.scene));
  }
});
test('a private background prepares and captures when all user scene materials use the fixed route', t => {
  const f = rig(t), fixed = f.mesh(new THREE.MeshBasicMaterial()), material = f.material(); let updates = 0;
  const background = new THREE.Mesh(new THREE.PlaneGeometry(), material);
  material.colorNode = N.uniform(.5).onObjectUpdate(() => { updates++; return .5; });
  f.scene.visible = false; background.layers.set(5);
  const children = f.scene.children.slice();
  const prepared = f.capture({backgroundMesh: background, uniforms: false});
  assert.equal(prepared.background, true); assert.equal(prepared.bindings.length, 1);
  assert.equal(prepared.bindings[0].object, background); assert.equal(prepared.draws.length, 0); assert.equal(updates, 0);
  const captured = f.capture({backgroundMesh: background});
  assert.equal(captured.background, true); assert.deepEqual(captured.objects, new Set([f.scene, fixed, background]));
  assert.deepEqual(captured.excludedObjects, new Set([f.scene, fixed]));
  assert.equal(captured.draws.length, 1); assert.equal(captured.draws[0].object, background);
  assert.equal(captured.draws[0].background, true); assert.equal(updates, 1);
  assert.deepEqual(f.scene.children, children); assert.equal(fixed.parent, f.scene); assert.equal(background.parent, null);
  const attached = new THREE.Group(); attached.add(background);
  assert.throws(() => f.capture({backgroundMesh: background}), {code: 'THREE_NODE_PROGRAM_BACKGROUND'});
});
test('native draws capture source identities, matrices, frame context and independent uniform histories', t => {
  const f = rig(t), seen = [], m = f.material(), tint = N.uniform(new THREE.Color());
  tint.onObjectUpdate(frame => { seen.push({object: frame.object, material: frame.material, camera: frame.camera,
    scene: frame.scene, renderer: frame.renderer, frameId: frame.frameId, renderId: frame.renderId,
    time: frame.time, deltaTime: frame.deltaTime}); return frame.object.userData.tint; });
  let after = 0; tint.updateAfterType = 'object'; tint.updateAfter = () => { after++; };
  m.colorNode = tint;
  const a = f.mesh(m), b = f.mesh(m); a.userData.tint = new THREE.Color(.125, .25, .375);
  b.userData.tint = new THREE.Color(.625, .75, .875); a.position.x = -.5; b.position.x = .5; b.scale.x = -1;
  const captured = f.capture();
  assert.equal(captured.bindings.length, 2); assert.equal(captured.draws.length, 2); assert.equal(after, 2);
  assert.equal(f.camera.coordinateSystem, THREE.WebGPUCoordinateSystem);
  assert.deepEqual([captured.draws[0].object, captured.draws[1].object], [a, b]);
  assert.ok(floats(captured.draws[0].nodeUniforms).includes(.125));
  assert.ok(floats(captured.draws[1].nodeUniforms).includes(.875));
  assert.equal(captured.draws[1].frontFaceCW, true); assert.equal(captured.draws[0].nodeRangeCaptured, true);
  assert.equal(a.modelViewMatrix.elements[14], -5); assert.equal(a.modelViewMatrix.elements[12], -.5);
  assert.equal(captured.bytes, captured.draws.reduce((sum, draw) => sum + draw.nodeUniforms.reduce((n, bytes) => n + bytes.byteLength, 0), 0));
  for (const [index, context] of seen.entries()) assert.deepEqual(context, {object: [a, b][index], material: m,
    camera: f.camera, scene: f.scene, renderer: f.renderer, frameId: 7, renderId: 3, time: 1.25, deltaTime: .125});
  const old = captured.draws.map(draw => draw.nodeUniforms.map(bytes => bytes.slice())), key = captured.bindings[0].nodeProgram;
  m.colorNode = N.vec3(1, 0, 0); m.transparent = true; a.renderOrder = 9; a.visible = false;
  const later = f.capture();
  assert.notEqual(later.bindings[0].nodeProgram, key); assert.equal(later.draws.length, 1);
  assert.equal(captured.draws[0].listMaterial.transparent, false); assert.equal(captured.draws[0].renderOrder, 0);
  assert.deepEqual(captured.draws.map(draw => draw.nodeUniforms), old); assert.equal(captured.draws.length, 2);
  assert.equal(f.frame.frameId, 7); assert.equal(f.renderer.info.calls, 3);
});
test('nested preparation preserves the outer render clock and source update frequency', t => {
  const f = rig(t), m = f.material(), seen = [];
  const observe = (phase, frame) => seen.push([phase, frame.frameId, frame.renderId, frame.time, frame.deltaTime]);
  const tint = N.uniform(.5).onRenderUpdate(frame => { observe('current', frame); return .5; });
  tint.updateBeforeType = 'render'; tint.updateBefore = frame => { observe('before', frame); };
  tint.updateAfterType = 'render'; tint.updateAfter = frame => { observe('after', frame); };
  m.colorNode = N.vec3(tint); f.mesh(m); f.mesh(m);
  let prepared = false;
  const programs = {...f.owner, compile(...args) {
    if (!prepared) {
      prepared = true;
      // A retained preparation helper may render before returning its program.
      // Its global call counters belong to those nested passes, not this draw.
      f.renderer.info.calls += 13;
      Object.assign(f.frame, {frameId: 8, time: 2, deltaTime: .25});
    }
    return f.owner.compile(...args);
  }};
  const captured = f.capture({getPrograms: () => programs});
  assert.equal(captured.draws.length, 2);
  assert.deepEqual(seen, ['before', 'current', 'after'].map(phase => [phase, 7, 3, 1.25, .125]));
  assert.equal(f.renderer.info.calls, 16, 'capturing the outer clock does not roll back nested counters');
  assert.deepEqual([f.frame.frameId, f.frame.time, f.frame.deltaTime], [8, 2, .25]);
});
test('compile-only capture excludes hidden bindings without constructing their programs or advancing updates', t => {
  const f = rig(t), m = f.material(); let calls = 0;
  m.colorNode = N.uniform(new THREE.Color()).onObjectUpdate(() => { calls++; return new THREE.Color(1, 0, 0); });
  const a = f.mesh(m), b = f.mesh(m); a.visible = false; b.visible = false;
  a.position.x = 3;
  const result = f.capture({uniforms: false, maxUniformBytes: 0, getPrograms() { assert.fail('inactive graphs must not compile'); }});
  assert.equal(result.objects.size, 2); assert.equal(result.bindings.length, 0);
  assert.deepEqual(result.excludedObjects, new Set([a, b]));
  assert.equal(result.draws.length, 0); assert.equal(result.bytes, 0); assert.equal(calls, 0);
  assert.equal(a.matrixWorld.elements[12], 3); assert.equal(f.camera.matrixWorld.elements[14], 5);
  a.visible = true;
  const visible = f.capture({uniforms: false, maxUniformBytes: 0});
  assert.equal(visible.bindings.length, 1); assert.equal(visible.bindings[0].object, a); assert.equal(calls, 0);
  assert.ok(result.excludedObjects.has(a), 'accepted visibility remains frozen after source flags change');
  assert.deepEqual([f.frame.frameId, f.frame.time, f.frame.deltaTime, f.renderer.info.calls], [7, 1.25, .125, 3]);
});
test('native Mesh.count is captured after updates, including empty draws, before after-updates', t => {
  const f = rig(t), m = f.material(), requested = [3, 0, -2, undefined], seen = [];
  const tint = N.uniform(.5).onObjectUpdate(frame => {
    frame.object.count = requested.shift(); seen.push('update'); return .5;
  });
  tint.updateAfterType = 'object'; tint.updateAfter = frame => { seen.push('after'); frame.object.count = 99; };
  m.colorNode = N.vec3(tint); m.positionNode = N.positionLocal.add(N.vec3(N.instanceIndex.toFloat().mul(.1), 0, 0));
  const object = f.mesh(m);
  const prepared = f.capture({uniforms: false}); assert.equal(seen.length, 0);
  const frames = Array.from({length: 4}, () => f.capture());
  assert.deepEqual(frames.map(frame => frame.draws[0].instanceCount), [3, 0, 0, 1]);
  assert.deepEqual(seen, ['update', 'after', 'update', 'after', 'update', 'after', 'update', 'after']);
  assert.equal(object.count, 99);
  assert.ok(frames.every(frame => frame.bindings[0].nodeProgram === prepared.bindings[0].nodeProgram));
  assert.match(prepared.bindings[0].nodeProgram.vertex, /@builtin\(\s*instance_index\s*\)/);
});

test('source selection excludes inherited hidden bindings and captures changing LOD visibility', t => {
  const f = rig(t), hidden = new THREE.Group(); hidden.visible = false; f.scene.add(hidden);
  const hiddenChild = f.mesh(undefined, hidden), offscreen = f.mesh(); offscreen.position.x = 100;
  const wrongLayer = f.mesh(); wrongLayer.layers.set(2);
  const differentLayerParent = new THREE.Group(); differentLayerParent.layers.set(2); f.scene.add(differentLayerParent);
  const visibleChild = f.mesh(undefined, differentLayerParent);
  const lod = new THREE.LOD(); f.scene.add(lod);
  const near = f.mesh(undefined, lod), far = f.mesh(undefined, lod); lod.addLevel(near, 0); lod.addLevel(far, 4);
  const result = f.capture();
  assert.equal(result.bindings.length, 4); assert.ok(result.objects.has(hiddenChild));
  assert.deepEqual(result.excludedObjects, new Set([hidden, hiddenChild, near]));
  assert.deepEqual(new Set(result.draws.map(draw => draw.object)), new Set([visibleChild, far]));
  assert.equal(near.visible, false); assert.equal(far.visible, true);
  f.camera.position.z = 2;
  const next = f.capture();
  assert.equal(next.bindings.length, 4);
  assert.deepEqual(new Set(next.draws.map(draw => draw.object)), new Set([visibleChild, near]));
  assert.deepEqual(next.excludedObjects, new Set([hidden, hiddenChild, far]));
  assert.equal(result.visibility.get(far), true); assert.equal(result.visibility.get(near), false);
});
test('native opaque sorting uses depth before object id, with source group and transparency order', t => {
  const f = rig(t), earlyMaterial = f.material(), lateMaterial = f.material();
  const farOpaque = f.mesh(earlyMaterial), nearOpaque = f.mesh(lateMaterial); nearOpaque.position.z = 1;
  const tm = f.material(); tm.transparent = true;
  const nearTransparent = f.mesh(tm), farTransparent = f.mesh(tm); nearTransparent.position.z = 1; farTransparent.position.z = -1;
  const group = new THREE.Group(); group.renderOrder = -2; f.scene.add(group); const ordered = f.mesh(undefined, group);
  assert.deepEqual(f.capture().draws.map(draw => draw.object), [ordered, nearOpaque, farOpaque, farTransparent, nearTransparent]);
  f.renderer.sortObjects = false;
  assert.deepEqual(f.capture().draws.map(draw => draw.object), [farOpaque, nearOpaque, ordered, nearTransparent, farTransparent]);
});
test('transparent native DoubleSide draws preserve source side order, identities and per-draw updates', t => {
  const f = rig(t), material = f.material(), seen = [];
  material.transparent = true; material.side = THREE.DoubleSide;
  const color = N.uniform(.5).onObjectUpdate(frame => {
    seen.push(['current', frame.object, frame.material, material.side]);
    return material.side === THREE.BackSide ? .25 : .75;
  });
  color.updateBeforeType = 'object'; color.updateBefore = frame => { seen.push(['before', frame.object, frame.material, material.side]); };
  color.updateAfterType = 'object'; color.updateAfter = frame => { seen.push(['after', frame.object, frame.material, material.side]); };
  material.colorNode = N.vec3(color);
  const near = f.mesh(material), far = f.mesh(material); near.position.z = 1; far.position.z = -1;
  const expected = [];
  for (const object of [far, near]) THREE.Renderer.prototype.renderObject.call({
    _handleObjectFunction(object, material, scene, camera, lights, group, clipping, passId) {
      expected.push([object, material.side, passId]);
    },
  }, object, f.scene, f.camera, object.geometry, material, null, null);
  const captured = f.capture();
  assert.deepEqual(captured.draws.map(draw => [draw.object, draw.binding.raster.side, draw.binding.passId]), expected);
  assert.deepEqual(seen, expected.flatMap(([object, side]) =>
    ['before', 'current', 'after'].map(phase => [phase, object, material, side])));
  assert.equal(material.side, THREE.DoubleSide);
  assert.equal(captured.bindings.length, 4); assert.equal(captured.draws.length, 4);
  for (const draw of captured.draws) {
    assert.equal(draw.material, material); assert.equal(draw.binding.material, material);
    assert.ok(floats(draw.nodeUniforms).includes(draw.binding.raster.side === THREE.BackSide ? .25 : .75));
  }
  assert.equal(captured.bytes, captured.draws.reduce((sum, draw) =>
    sum + draw.nodeUniforms.reduce((n, bytes) => n + bytes.byteLength, 0), 0));
  const saved = captured.draws.map(draw => draw.nodeUniforms.map(bytes => bytes.slice()));
  material.side = THREE.FrontSide; material.forceSinglePass = true; f.renderer.info.calls++;
  assert.equal(f.capture().draws.length, 2);
  assert.deepEqual(captured.draws.map(draw => draw.nodeUniforms), saved);
  assert.deepEqual(captured.draws.map(draw => draw.binding.raster.side), [THREE.BackSide, THREE.FrontSide, THREE.BackSide, THREE.FrontSide]);
});
test('native transparent side variants retain generated streams and reuse both compiled programs', t => {
  const f = rig(t), material = f.material(); material.transparent = true; material.side = THREE.DoubleSide;
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  material.positionNode = N.bufferAttribute(positions, 'vec3');
  f.mesh(material);
  const first = f.capture(), descriptors = first.bindings.map(binding => binding.nodeProgram);
  assert.equal(new Set(descriptors).size, 2); assert.equal(f.owner.diagnostics.programs, 2);
  assert.ok(first.draws.every(draw => draw.nodeAttributes));
  assert.doesNotThrow(() => first.installAttributes());
  f.renderer.info.calls++;
  const second = f.capture();
  assert.deepEqual(second.bindings.map(binding => binding.nodeProgram), descriptors);
  assert.equal(f.owner.diagnostics.programs, 2);
  assert.doesNotThrow(() => first.installAttributes());
  assert.doesNotThrow(() => second.installAttributes());
  material.dispose(); assert.equal(f.owner.diagnostics.programs, 0);
});
test('transparent side compilation is callback-free and preserves source frame/render update frequencies', t => {
  const f = rig(t), material = f.material(), counts = {frame: 0, render: 0, object: 0};
  material.transparent = true; material.side = THREE.DoubleSide;
  material.colorNode = N.vec3(
    N.uniform(.25).onFrameUpdate(() => { counts.frame++; return .25; }),
    N.uniform(.5).onRenderUpdate(() => { counts.render++; return .5; }),
    N.uniform(.75).onObjectUpdate(() => { counts.object++; return .75; }));
  f.mesh(material); f.mesh(material);
  const prepared = f.capture({uniforms: false});
  assert.equal(prepared.bindings.length, 4); assert.equal(prepared.draws.length, 0);
  assert.deepEqual(counts, {frame: 0, render: 0, object: 0});
  f.capture(); assert.deepEqual(counts, {frame: 1, render: 1, object: 4});
  f.renderer.info.calls++; f.capture(); assert.deepEqual(counts, {frame: 1, render: 2, object: 8});
  f.frame.frameId++; f.renderer.info.calls++; f.capture(); assert.deepEqual(counts, {frame: 2, render: 3, object: 12});
  assert.equal(material.side, THREE.DoubleSide);
});
test('native transparent variants respect forceSinglePass, opacity and expanded capacity before updates', t => {
  for (const [transparent, side, forceSinglePass, expected] of [
    [true, THREE.DoubleSide, false, [THREE.BackSide, THREE.FrontSide]],
    [true, THREE.DoubleSide, true, [THREE.DoubleSide]],
    [false, THREE.DoubleSide, false, [THREE.DoubleSide]],
    [true, THREE.BackSide, false, [THREE.BackSide]],
  ]) {
    const f = rig(t), material = f.material(); Object.assign(material, {transparent, side, forceSinglePass});
    f.mesh(material);
    assert.deepEqual(f.capture().draws.map(draw => draw.binding.raster.side), expected);
    assert.equal(material.side, side);
  }
  const f = rig(t), material = f.material(); let updates = 0;
  material.transparent = true; material.side = THREE.DoubleSide;
  material.colorNode = N.uniform(.5).onObjectUpdate(() => { updates++; return .5; });
  const object = f.mesh(material); object.geometry.clearGroups();
  object.geometry.addGroup(0, 3, 0); object.geometry.addGroup(3, 3, 0); object.material = [material];
  assert.throws(() => f.capture({maxDraws: 3}), {code: 'THREE_NODE_PROGRAM_LIMIT'});
  assert.equal(updates, 0); assert.equal(material.side, THREE.DoubleSide);
  assert.equal(f.capture({maxDraws: 4}).draws.length, 4); assert.equal(updates, 4);
});
test('native transparent side state is restored when a later draw update fails', t => {
  const f = rig(t), material = f.material(); material.transparent = true; material.side = THREE.DoubleSide;
  material.colorNode = N.uniform(.5).onObjectUpdate(() => {
    if (material.side === THREE.FrontSide) throw new Error('front-side update failure');
    return .25;
  });
  f.mesh(material);
  assert.throws(f.capture, /front-side update failure/);
  assert.equal(material.side, THREE.DoubleSide);
});
test('material groups capture their intersections and update once per logical draw', t => {
  const f = rig(t), a = f.material(), b = f.material(); let updates = 0;
  const value = N.uniform(.25).onObjectUpdate(() => { updates++; return .25; }); a.colorNode = N.vec3(value); b.colorNode = N.vec3(value);
  const object = f.mesh([a, b]), geometry = object.geometry; geometry.clearGroups();
  geometry.addGroup(0, 3, 0); geometry.addGroup(3, 3, 1); geometry.addGroup(1, 3, 0); geometry.setDrawRange(1, 4);
  const result = f.capture();
  assert.equal(result.bindings.length, 2); assert.equal(updates, 3);
  assert.deepEqual(result.draws.map(draw => [draw.first, draw.count]), [[1, 2], [3, 2], [1, 3]]);
  geometry.groups[0].start = 5; geometry.setDrawRange(4, 1);
  assert.deepEqual(result.draws.map(draw => [draw.first, draw.count]), [[1, 2], [3, 2], [1, 3]]);
  assert.equal(result.draws[0].group.start, 0); assert.equal(result.draws[0].binding, result.draws[2].binding);
});
test('mixed built-in and authored material groups share native geometry and retain identities', t => {
  const f = rig(t), authored = f.material(), ordinary = new THREE.MeshBasicMaterial({color: 0x4488cc});
  const object = f.mesh([authored, ordinary]);
  object.geometry.clearGroups(); object.geometry.addGroup(0, 3, 0); object.geometry.addGroup(3, 3, 1);
  const result = f.capture();
  assert.deepEqual(result.bindings.map(binding => binding.material), [authored, ordinary]);
  assert.deepEqual(result.draws.map(draw => [draw.material, draw.first, draw.count]), [[authored, 0, 3], [ordinary, 3, 3]]);
  assert.ok(result.draws.every(draw => draw.object === object && draw.geometry === object.geometry));
  assert.equal(ordinary.isNodeMaterial, undefined, 'source material identity is not replaced by conversion');
});
test('active invalid graphs, unsupported material groups and effectful upload/object hooks refuse before updates', t => {
  const cases = [
    f => { const m = f.material(); m.colorNode = N.texture(new THREE.Texture()); f.mesh(m); },
    f => { f.mesh([f.material(), new THREE.ShaderMaterial()]); },
    f => { f.mesh().geometry.attributes.position.onUpload(() => {}); },
    f => { f.mesh().geometry.index.onUpload(() => {}); },
    f => { const o = f.mesh(); const data = new THREE.InterleavedBuffer(new Float32Array(12), 3); data.onUpload(() => {});
      o.geometry.setAttribute('position', new THREE.InterleavedBufferAttribute(data, 3, 0)); },
    f => { f.mesh().onBeforeRender = () => {}; },
    f => { f.mesh().intersectsFrustum = () => true; },
  ];
  for (const bad of cases) {
    const f = rig(t), m = f.material(); let updates = 0;
    m.colorNode = N.uniform(.5).onObjectUpdate(() => { updates++; return .5; }); f.mesh(m); bad(f);
    assert.throws(() => f.capture(), error => error.code?.startsWith('THREE_NODE_PROGRAM_'));
    assert.equal(updates, 0);
  }
});
test('hidden authored graphs stay inactive through their ancestor and re-admit before any draw update', t => {
  const f = rig(t), activeMaterial = f.material(), hidden = new THREE.Group(); let updates = 0;
  activeMaterial.colorNode = N.uniform(.5).onObjectUpdate(() => { updates++; return .5; }); f.mesh(activeMaterial);
  hidden.visible = false; f.scene.add(hidden);
  const invalidMaterial = f.material(); invalidMaterial.colorNode = N.texture(new THREE.Texture());
  const child = f.mesh(invalidMaterial, hidden);
  const accepted = f.capture();
  assert.equal(accepted.bindings.length, 1); assert.equal(updates, 1);
  assert.ok(accepted.excludedObjects.has(child)); assert.equal(child.visible, true);
  hidden.visible = true;
  assert.throws(() => f.capture(), error => error.code?.startsWith('THREE_NODE_PROGRAM_'));
  assert.equal(updates, 1); assert.equal(accepted.visibility.get(child), false);
});
test('all compile and raster admission completes before callbacks, and budgets preflight without partial updates', t => {
  const f = rig(t), m = f.material(); let updates = 0;
  m.colorNode = N.uniform(.5).onObjectUpdate(() => { updates++; return .5; }); f.mesh(m); f.mesh(m);
  const compiled = f.capture({uniforms: false, maxUniformBytes: 0});
  const bytes = compiled.bindings.reduce((sum, binding) => sum + binding.nodeProgram.groups.reduce((n, group) => n + group.byteLength, 0), 0);
  assert.throws(() => f.capture({maxUniformBytes: bytes - 1}), {code: 'THREE_NODE_PROGRAM_LIMIT'});
  assert.equal(updates, 0);
  let rasterCalls = 0;
  assert.throws(() => f.capture({raster() { if (++rasterCalls === 2) throw Error('second raster refuses'); return {}; }}), /second raster refuses/);
  assert.equal(updates, 0);
  assert.throws(() => f.capture({maxDraws: 1}), {code: 'THREE_NODE_PROGRAM_LIMIT'}); assert.equal(updates, 0);
  assert.equal(f.capture({maxUniformBytes: bytes}).bytes, bytes); assert.equal(updates, 2);
});
test('equivalent overrides preserve source allowOverride, original visibility and callback material identities', t => {
  const f = rig(t), original = f.material(), override = f.material(), excluded = f.material(); excluded.allowOverride = false;
  const a = f.mesh(original), b = f.mesh(excluded); override.visible = false;
  let selected = null; override.colorNode = N.uniform(.25).onObjectUpdate(frame => { selected = frame.material; return .25; });
  const captured = f.capture({overrideMaterial: override});
  assert.deepEqual(captured.draws.map(draw => draw.material), [override, excluded]); assert.equal(selected, override);
  original.visible = false;
  assert.deepEqual(f.capture({overrideMaterial: override}).draws.map(draw => draw.object), [b]);
  original.visible = true; original.positionNode = N.positionLocal.add(N.vec3(1, 0, 0));
  assert.throws(() => f.capture({overrideMaterial: override}), {code: 'THREE_NODE_PROGRAM_EFFECT'});
  assert.equal(override.positionNode, null); assert.equal(a.material, original);
});
test('capture freezes the render context across callbacks that mutate renderer counters', t => {
  const f = rig(t), seen = [], m = f.material();
  m.colorNode = N.uniform(.25).onObjectUpdate(frame => { seen.push([frame.frameId, frame.renderId, frame.time]);
    f.renderer.info.calls = 91; return .25; }); f.mesh(m); f.mesh(m);
  f.capture(); assert.deepEqual(seen, [[7, 3, 1.25], [7, 3, 1.25]]);
});
test('capture bounds cyclic hierarchies and hidden node growth before matrix recursion', t => {
  const f = rig(t), object = f.mesh();
  assert.throws(() => f.capture({maxNodes: 1}), {code: 'THREE_NODE_PROGRAM_LIMIT'});
  object.children.push(f.scene);
  assert.throws(() => f.capture(), {code: 'THREE_NODE_PROGRAM_GRAPH'});
});
test('storage-only compile capture needs no GPU publication and runs no node callbacks', t => {
  const f = rig(t), device = geometryDevice(); f.renderer._device = device;
  const attribute = new THREE.StorageBufferAttribute(4, 4), storage = N.storage(attribute, 'vec4', 4);
  let updates = 0, transactions = 0;
  const gain = N.uniform(1).onObjectUpdate(() => { updates++; return 1; });
  const material = f.material(); material.fragmentNode = storage.element(0).mul(gain); f.mesh(material);
  const guard = {captureDispatch(run) { transactions++; return run(); }, validate() {}};
  const result = f.capture({uniforms: false, getStorage: () => guard});
  assert.equal(result.bindings.length, 1); assert.equal(result.draws.length, 0);
  assert.equal(f.owner.storage(result.bindings[0].nodeProgram)[0].attribute, attribute);
  assert.equal(updates, 0); assert.equal(device.buffers.length, 0); assert.equal(device.writes.length, 0);
  assert.equal(result.storageOwner, guard, 'storage-only graphs join the shared capture transaction');
  assert.equal(transactions, 1);
});
test('storage captures lease post-update identities before later draws and after-updates can replace them', t => {
  const f = rig(t), device = geometryDevice(), gpu = createGpuCompute(device); f.renderer._device = device;
  t.after(() => gpu.dispose());
  const a = new THREE.StorageBufferAttribute(4, 4), b = new THREE.StorageBufferAttribute(4, 4),
    after = new THREE.StorageBufferAttribute(4, 4), storage = N.storage(a, 'vec4', 4);
  a.array.fill(.125); b.array.fill(.875);
  const aBuffer = gpu.upload(a, a.array, a.version), bBuffer = gpu.upload(b, b.array, b.version);
  const material = f.material(), first = f.mesh(material), second = f.mesh(material);
  let inside = false, transactions = 0, updates = 0;
  const guard = {captureDispatch(run) {
    assert.equal(inside, false); transactions++; inside = true;
    try { return run(); } finally { inside = false; }
  }, capture: attribute => captureComputeStorage(device, attribute), validate() {}, release() {}};
  const gain = N.uniform(1).onObjectUpdate(frame => {
    assert.equal(inside, true, 'storage callbacks run inside the shared capture transaction');
    updates++; storage.value = frame.object === first ? a : b; return 1;
  });
  gain.updateAfterType = 'object'; gain.updateAfter = () => { storage.value = after; };
  material.fragmentNode = storage.element(0).mul(gain);
  const writes = device.writes.length, result = f.capture({getStorage: () => guard});
  assert.equal(updates, 2); assert.equal(transactions, 2, 'each draw captures after its source before-updates'); assert.equal(storage.value, after);
  assert.deepEqual(result.draws.map(draw => draw.object), [first, second]);
  assert.deepEqual(result.draws.map(draw => draw.nodeStorage[0].lease.resolve().buffer), [aBuffer, bBuffer]);
  assert.ok(result.draws.every(draw => Object.isFrozen(draw.nodeStorage) && draw.nodeStorage.every(Object.isFrozen)));
  assert.equal(result.textureUses.length, 0); assert.equal(device.writes.length, writes, 'draw capture never uploads storage');
  a.array.fill(99); b.array.fill(99);
  assert.deepEqual(result.draws.map(draw => Array.from(new Float32Array(draw.nodeStorage[0].lease.resolve().buffer.data)).slice(0, 4)),
    [[.125, .125, .125, .125], [.875, .875, .875, .875]]);
});

test('per-draw capture releases earlier leases when a later source update fails', t => {
  const f = rig(t), material = f.material(), first = f.mesh(material), second = f.mesh(material);
  const storage = N.storage(new THREE.StorageBufferAttribute(4, 4), 'vec4', 4);
  const leases = [], released = [];
  let inside = false;
  const guard = {captureDispatch(run) {
    assert.equal(inside, false); inside = true;
    try { return run(); } finally { inside = false; }
  }, capture(attribute) { const lease = {attribute}; leases.push(lease); return lease; },
  release(lease) { released.push(lease); }, validate() { assert.fail('failed capture cannot validate'); }};
  const gain = N.uniform(1).onObjectUpdate(() => 1);
  gain.updateAfterType = 'object'; gain.updateAfter = frame => {
    assert.equal(inside, true);
    if (frame.object === second) throw new Error('late shadow draw failure');
  };
  material.fragmentNode = storage.element(0).mul(gain);
  assert.throws(() => f.capture({getStorage: () => guard}), /late shadow draw failure/);
  assert.equal(leases.length, 2); assert.deepEqual(released, leases); assert.equal(inside, false);
  assert.equal(f.frame.object, null); assert.equal(first.material, material);
});

test('source before-updates can capture child draws outside the parent resource transaction', t => {
  const f = rig(t), material = f.material(), storage = N.storage(new THREE.StorageBufferAttribute(4, 4), 'vec4', 4);
  material.fragmentNode = storage.element(0); f.mesh(material);
  let inside = false, nested = false, transactions = 0;
  const guard = {captureDispatch(run) {
    assert.equal(inside, false); inside = true; transactions++;
    try { return run(); } finally { inside = false; }
  }, capture(attribute) { return {attribute}; }, release() {}, validate() {}};
  const programs = {...f.owner, before(descriptor, context) {
    assert.equal(inside, false);
    if (!nested) { nested = true; f.capture({getStorage: () => guard}); }
    f.owner.before?.(descriptor, context);
  }};
  const result = f.capture({getPrograms: () => programs, getStorage: () => guard});
  assert.equal(result.draws.length, 1); assert.equal(nested, true); assert.equal(transactions, 2);
  assert.equal(inside, false);
});

test('nested writes reject an earlier parent depth read, and release the guard after failure', () => {
  const renderer = {}, target = new THREE.RenderTarget(8, 8);
  target.depthTexture = new THREE.DepthTexture(8, 8);
  const unrelated = new THREE.RenderTarget(8, 8);
  assert.throws(() => withThreeNativeCapture(renderer, record => {
    assert.doesNotThrow(() => assertThreeNativeTargetWrite(renderer, target));
    record(target.depthTexture);
    withThreeNativeCapture(renderer, recordChild => {
      recordChild(unrelated.texture);
      assert.throws(() => assertThreeNativeTargetWrite(renderer, target), {code: 'THREE_NODE_PROGRAM_SHADOW'});
      assert.throws(() => assertThreeNativeTargetWrite(renderer, unrelated), {code: 'THREE_NODE_PROGRAM_SHADOW'});
      throw new Error('nested capture failed');
    });
  }), /nested capture failed/);
  assert.doesNotThrow(() => assertThreeNativeTargetWrite(renderer, target));
  assert.doesNotThrow(() => assertThreeNativeTargetWrite(renderer, unrelated));
});

test('compute reentry stays refused until the complete native capture unwinds', () => {
  const renderer = {};
  assert.doesNotThrow(() => assertThreeNativeComputeOutsideCapture(renderer));
  withThreeNativeCapture(renderer, () => {
    assert.throws(() => assertThreeNativeComputeOutsideCapture(renderer), {code: 'THREE_COMPUTE_REENTRY'});
    withThreeNativeCapture(renderer, () =>
      assert.throws(() => assertThreeNativeComputeOutsideCapture(renderer), {code: 'THREE_COMPUTE_REENTRY'}));
    assert.throws(() => assertThreeNativeComputeOutsideCapture(renderer), {code: 'THREE_COMPUTE_REENTRY'});
  });
  assert.doesNotThrow(() => assertThreeNativeComputeOutsideCapture(renderer));
});
