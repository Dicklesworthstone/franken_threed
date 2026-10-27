import test from 'node:test';
import assert from 'node:assert/strict';
import * as T from '../../upstream/three.js/build/three.webgpu.js';
import {createThreeNodePrograms} from './three_node_program.mjs';

const {NONE, SHARED, FULL} = T.RenderObjectRefreshType;
function fixture(t, {material = new T.MeshBasicNodeMaterial(), instances = null, dynamic = false} = {}) {
  const renderer = {library: new T.StandardNodeLibrary(), contextNode: T.TSL.context(),
    debug: {diagnostics: {keywords: false}}, depth: true, lighting: {enabled: true},
    backend: {hasFeature: () => false, utils: {getTextureSampleData: () => ({samples: 1, primarySamples: 1, isMSAA: false})},
      capabilities: {getUniformBufferLimit: () => 65536}},
    hasFeature: () => false, getRenderTarget: () => null, getMRT: () => null,
    currentSamples: 1, _nodes: {nodeFrame: new T.NodeFrame()}, info: {calls: 1}};
  const owner = createThreeNodePrograms({three: T, renderer}); t.after(() => owner.dispose());
  const geometry = new T.PlaneGeometry(), object = instances === null ? new T.Mesh(geometry, material)
    : new T.InstancedMesh(geometry, material, instances);
  if (dynamic) object.instanceMatrix.setUsage(T.DynamicDrawUsage);
  const scene = new T.Scene(), bundle = new T.BundleGroup(); scene.add(bundle); bundle.add(object);
  const camera = new T.PerspectiveCamera(60, 1, .1, 100); camera.coordinateSystem = T.WebGPUCoordinateSystem;
  camera.position.z = 4; camera.updateProjectionMatrix(); scene.updateMatrixWorld(); camera.updateMatrixWorld();
  t.after(() => { geometry.dispose(); material.dispose(); });
  const context = {object, material, scene, camera, renderId: 1, frameId: 1, lights: [],
    bundle, bundleToken: {}, renderContext: {width: 32, height: 32}};
  const descriptor = owner.compile(material, object, context);
  const draw = (renderId, token = context.bundleToken, extra = {}) => {
    scene.updateMatrixWorld(); camera.updateMatrixWorld();
    const current = {...context, renderId, bundleToken: token, ...extra};
    current.refreshType = owner.refresh(descriptor, current);
    owner.before(descriptor, current);
    const packet = owner.captureDraw(descriptor, {...current, beforeDone: true});
    owner.after(descriptor, current);
    return packet;
  };
  const bytes = (packet, name) => packet.uniforms[descriptor.groups.findIndex(group => group.name === name)];
  return {owner, renderer, object, geometry, material, scene, camera, bundle, context, descriptor, draw, bytes};
}

test('static bundle local uniforms stay frozen while the camera group remains current', t => {
  const f = fixture(t), initial = f.draw(1);
  assert.equal(initial.refreshType, FULL);
  const oldLocal = f.bytes(initial, 'object').slice(), oldCamera = f.bytes(initial, 'render').slice();
  f.object.position.x = 3; f.material.color.setRGB(.2, .4, .6); f.camera.position.x = 2;
  const warm = f.draw(2);
  assert.equal(warm.refreshType, SHARED);
  assert.deepEqual(f.bytes(warm, 'object'), oldLocal);
  assert.notDeepEqual(f.bytes(warm, 'render'), oldCamera);
  assert.deepEqual(f.bytes(initial, 'render'), oldCamera, 'an earlier accepted render owns its bytes');
  f.bundle.static = false;
  const dynamic = f.draw(3);
  assert.equal(dynamic.refreshType, FULL);
  assert.notDeepEqual(f.bytes(dynamic, 'object'), oldLocal);
  f.bundle.static = true; f.object.position.x = 7;
  const paused = f.draw(4);
  assert.equal(paused.refreshType, SHARED);
  assert.deepEqual(f.bytes(paused, 'object'), f.bytes(dynamic, 'object'));
});

test('source needsUpdate/version inequality refreshes static bundles without a monotonic-version assumption', t => {
  const f = fixture(t), initial = f.draw(1);
  f.material.color.setRGB(.1, .2, .3); f.bundle.needsUpdate = true;
  const changed = f.draw(2);
  assert.equal(changed.refreshType, FULL);
  assert.notDeepEqual(f.bytes(changed, 'object'), f.bytes(initial, 'object'));
  // The source observer synchronizes changed fields incrementally, then its
  // bundle version. Stabilize before testing a version decrement.
  f.draw(3); f.draw(4);
  f.material.color.setRGB(.7, .6, .5); f.bundle.version = 0;
  const revertedVersion = f.draw(5);
  assert.equal(revertedVersion.refreshType, FULL);
  assert.notDeepEqual(f.bytes(revertedVersion, 'object'), f.bytes(changed, 'object'));
});

test('NONE does not replay node updates and observes shared bytes refreshed by another render object', t => {
  const f = fixture(t), other = {};
  const first = f.draw(1);
  f.material.color.setRGB(.2, .4, .6); const second = f.draw(1, other);
  assert.notDeepEqual(f.bytes(first, 'object'), f.bytes(second, 'object'));
  f.camera.position.x = 2;
  const warm = f.draw(2);
  let updates = 0;
  const frame = f.renderer._nodes.nodeFrame, update = frame.updateNode;
  frame.updateNode = function(node) { updates++; return update.call(this, node); };
  const unchanged = f.draw(2, other);
  assert.equal(unchanged.refreshType, NONE); assert.equal(updates, 0);
  assert.deepEqual(f.bytes(unchanged, 'object'), f.bytes(second, 'object'));
  assert.deepEqual(f.bytes(unchanged, 'render'), f.bytes(warm, 'render'));
});

test('shared refresh retains the original texture selection and does not upload new instance data', t => {
  const original = new T.DataTexture(new Uint8Array([255, 0, 0, 255]), 1, 1);
  const replacement = new T.DataTexture(new Uint8Array([0, 255, 0, 255]), 1, 1);
  t.after(() => { original.dispose(); replacement.dispose(); });
  const material = new T.MeshBasicNodeMaterial({map: original});
  const f = fixture(t, {material, instances: 2}), initial = f.draw(1);
  material.map = replacement;
  f.object.setMatrixAt(0, new T.Matrix4().makeTranslation(5, 0, 0)); f.object.instanceMatrix.needsUpdate = true;
  const warm = f.draw(2);
  assert.equal(warm.refreshType, SHARED);
  assert.equal(warm.textures, initial.textures); assert.equal(warm.storage, initial.storage);
  assert.equal(warm.attributes, initial.attributes);
  assert.ok(warm.textures.every(binding => binding.texture === original));
  assert.deepEqual(f.bytes(warm, 'object'), f.bytes(initial, 'object'));
});

test('dynamic instance usage requires FULL refresh even inside a static bundle', t => {
  const f = fixture(t, {instances: 2, dynamic: true}), first = f.draw(1);
  f.object.setMatrixAt(0, new T.Matrix4().makeTranslation(3, 0, 0));
  const second = f.draw(2);
  assert.equal(second.refreshType, FULL);
  assert.notDeepEqual(f.bytes(second, 'object'), f.bytes(first, 'object'));
});

test('authored node updates remain FULL and bundle geometry is captured before ordinary node effects', t => {
  let object, calls = 0;
  const change = T.TSL.uniform(0).onObjectUpdate(() => {
    calls++; object.geometry.attributes.position.array[0] = calls * 10;
    object.geometry.attributes.position.needsUpdate = true; return 0;
  });
  const material = new T.MeshBasicNodeMaterial(); material.positionNode = T.TSL.positionLocal.add(T.TSL.vec3(change, 0, 0));
  const f = fixture(t, {material}); object = f.object;
  const original = f.geometry.attributes.position.array[0];
  const captureGeometry = view => ({first: view.attributes.position.array[0]});
  const first = f.draw(1, f.context.bundleToken, {captureGeometry});
  const second = f.draw(2, f.context.bundleToken, {captureGeometry});
  assert.equal(first.refreshType, FULL); assert.equal(second.refreshType, FULL); assert.equal(calls, 2);
  assert.equal(first.bundleGeometry.first, original);
  assert.equal(second.bundleGeometry.first, 10);
  assert.equal(f.geometry.attributes.position.array[0], 20);
});

test('geometry captures are retained by SHARED and NONE without recapturing live source arrays', t => {
  const f = fixture(t); let captures = 0;
  const captureGeometry = view => { captures++; return {positions: view.attributes.position.array.slice()}; };
  const initial = f.draw(1, f.context.bundleToken, {captureGeometry});
  f.geometry.attributes.position.array.fill(7); f.geometry.attributes.position.needsUpdate = true;
  const shared = f.draw(2, f.context.bundleToken, {captureGeometry});
  const none = f.draw(2, f.context.bundleToken, {captureGeometry});
  assert.equal(shared.refreshType, SHARED); assert.equal(none.refreshType, NONE); assert.equal(captures, 1);
  assert.equal(shared.bundleGeometry, initial.bundleGeometry); assert.equal(none.bundleGeometry, initial.bundleGeometry);
});

test('bundle records cannot be reused across groups and shared capture cannot precede source initialization', t => {
  const f = fixture(t);
  assert.throws(() => f.owner.captureDraw(f.descriptor, {...f.context, refreshType: SHARED}), {code: 'THREE_NODE_PROGRAM_SOURCE'});
  f.draw(1);
  assert.throws(() => f.owner.refresh(f.descriptor, {...f.context, bundle: new T.BundleGroup()}), {code: 'THREE_NODE_PROGRAM_SOURCE'});
  assert.throws(() => f.owner.refresh(f.descriptor, {...f.context, bundleToken: null}), {code: 'THREE_NODE_PROGRAM_SOURCE'});
});

test('FULL refresh keeps the recorded rigid geometry and material after source property replacement', t => {
  const f = fixture(t); f.bundle.static = false;
  const captureGeometry = geometry => ({geometry});
  const initial = f.draw(1, f.context.bundleToken, {captureGeometry});
  const nextGeometry = new T.BoxGeometry(), nextMaterial = new T.MeshBasicNodeMaterial({color: 0xff0000});
  t.after(() => { nextGeometry.dispose(); nextMaterial.dispose(); });
  f.object.geometry = nextGeometry; f.object.material = nextMaterial; f.object.position.x = 2;
  const warm = f.draw(2, f.context.bundleToken, {captureGeometry});
  assert.equal(warm.refreshType, FULL); assert.equal(warm.bundleGeometry.geometry, initial.bundleGeometry.geometry);
  assert.equal(warm.bundleGeometry.geometry, f.geometry);
  assert.equal(f.object.geometry, nextGeometry); assert.equal(f.object.material, nextMaterial);
});

test('geometry initial leases retire on successful FULL replacement and owner retirement', t => {
  const f = fixture(t), released = [], captureGeometry = () => {
    const id = ++created; let done = false;
    return {id, release() { if (!done) { done = true; released.push(id); } }};
  };
  let created = 0;
  f.bundle.static = false;
  f.draw(1, f.context.bundleToken, {captureGeometry});
  f.object.position.x = 1; f.draw(2, f.context.bundleToken, {captureGeometry});
  assert.deepEqual(released, [1]);
  f.owner.release(f.descriptor); assert.deepEqual(released, [1, 2]);
  // Accepted descriptors can outlive their program-cache entry. A later
  // retained bundle refresh still has its geometry released at owner disposal.
  f.object.position.x = 2; f.draw(3, f.context.bundleToken, {captureGeometry});
  f.owner.dispose(); assert.deepEqual(released, [1, 2, 3]);
});

test('a failing node update releases the newly captured geometry without retiring accepted history', t => {
  const failure = new Error('source node update'), released = []; let reject = false, created = 0;
  const node = T.TSL.uniform(0).onObjectUpdate(() => { if (reject) throw failure; return 0; });
  const material = new T.MeshBasicNodeMaterial(); material.positionNode = T.TSL.positionLocal.add(T.TSL.vec3(node, 0, 0));
  const f = fixture(t, {material}), captureGeometry = () => {
    const id = ++created; let done = false;
    return {id, release() { if (!done) { done = true; released.push(id); } }};
  };
  const accepted = f.draw(1, f.context.bundleToken, {captureGeometry}); reject = true;
  assert.throws(() => f.draw(2, f.context.bundleToken, {captureGeometry}), error => error === failure);
  assert.equal(accepted.bundleGeometry.id, 1); assert.deepEqual(released, [2]);
  f.owner.dispose(); assert.deepEqual(released, [2, 1]);
});
